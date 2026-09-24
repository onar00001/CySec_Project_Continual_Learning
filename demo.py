import streamlit as st
import torch
from CNND_model import Model_CNND
import os
import torchvision.transforms as transforms
from PIL import Image
from argparse import Namespace
import pandas as pd
import numpy as np
import gc
import sys
import models_lucir.modified_resnet
from models_lucir.modified_resnet import resnet50
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # fixed module not found error

sys.modules['models'] = sys.modules.get('models_lucir')
sys.modules['models.modified_resnet'] = models_lucir.modified_resnet

task_dict = {1:"GauGAN", 2:"BigGAN", 3:"imle", 4:"deepfake" , 5:"wild"}

checkpoint_dir= "./checkpoints_with_only_5_tasks/"
checkpoint_naive_dir = "./checkpoints_naive_with_only_5_tasks/"
checkpoint_naive_dir_l ="./checkpoints_naive_l"
checkpoint_dir_l = "./checkpoints_lucir"

total_eval_steps = 25

Transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

def page_title_config():
    st.set_page_config(page_title="Continual vs Common Deepfake Detection",layout="wide")

@st.cache_resource(max_entries=1)
def load_naive_model(state_idx):
    file_name = f"naive_model_state_{state_idx}_after_t{state_idx-1}.pt"
    path = os.path.join(checkpoint_naive_dir,file_name)

    if not os.path.exists(path):
        return None
    
    demo_args = Namespace(model_name='Demo', model_weights=None)
    model = Model_CNND(1,args=demo_args)
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_resource(max_entries=1)
def load_naive_model_luc(state_idx): # resnet50 with cosinelinear layer in contrast to naive flag that was set for icarl code
    file_name = f"naive_model_state_{state_idx}_after_t{state_idx-1}.pt"
    path = os.path.join(checkpoint_naive_dir_l,file_name)
    if not os.path.exists(path):
            return None
    model = torch.load(path,map_location=torch.device('cpu'),weights_only=False)
    model.eval()
    return model

@st.cache_resource(max_entries=1)
def load_lucir_model(state_idx): 
    file_name = f"model_state_{state_idx}_after_t{state_idx-1}.pt"
    path = os.path.join(checkpoint_dir_l,file_name)
    if not os.path.exists(path):
            return None
    model = torch.load(path,map_location=torch.device('cpu'),weights_only=False)
    model.eval()
    return model
    

@st.cache_resource(max_entries=1)
def load_model_and_means(state_idx,total_tasks=5):
    file_name = f"model_state_{state_idx}_after_t{state_idx-1}.pt"
    path = os.path.join(checkpoint_dir,file_name)
    if not os.path.exists(path):
        return None,None
    demo_args = Namespace(model_name='Demo', model_weights=None)
    model = Model_CNND(1,args=demo_args)
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
# Class means (Exemplars) laden für iCaRL für state > 0
    class_means = None
    if state_idx > 0:
        means_file_name = f"cl_means_incr{state_idx}_of_{total_tasks}.pt"
        means_path = os.path.join(checkpoint_dir,means_file_name)
        if os.path.exists(means_path):
            class_means = torch.load(means_path,map_location=torch.device('cpu'))
        else:
            st.error(f"Class Means file not found :{means_file_name}")
    return model,class_means
    

def naive_inference(model,img_tensor): # done directly on model which just has logit output (prob of being fake)
    with torch.no_grad():
        output = model(img_tensor)
        pred_prob = output.item()
        is_fake = pred_prob > 0.5
        confidence = pred_prob if is_fake else (1-pred_prob)
        return is_fake,confidence 

def icarl_inference(model,class_means,img_tensor): # we use class means which has dimension d, 2 x tasks
    if class_means is None:
        return None,None

    with torch.no_grad():
        # feature extraction
        features = model.feature_extractor(img_tensor)   

        # L2 normalisation
        pred_inter = (features.T / torch.norm(features.T,dim=0)).T

        # calculate distance to stored class means
        sqd = torch.cdist(class_means[:,:,0].T,pred_inter)
        score_icarl = (-sqd).T

        # Classification
        best_class = torch.argmax(score_icarl,dim=1).item()
        probs = torch.softmax(score_icarl,dim=1)
       

        # indices contain probability being a (1-GauGAN fake, 2-BigGAN fake etc)
        # fake_prob contains prob that it is just fake
        fake_class_idx = [i for i in range(score_icarl.shape[1]) if i % 2 == 1] 
        fake_prob = sum([probs[0,idx].item() for idx in fake_class_idx])

        is_fake = fake_prob > 0.5
        confidence = fake_prob if is_fake else 1.0 - fake_prob

        return is_fake,confidence

def lucir_inference(model,img_tensor): # number of classes that model can classify is 2 (real or fake , logits displayed in array) 
    with torch.no_grad():
        output = model(img_tensor)
        logits = output['logits']
        probs = torch.softmax(logits,dim=1)
        fake_class_idx = [i for i in range(logits.shape[1]) if i%2 == 1]
        fake_prob = sum([probs[0,idx].item() for idx in fake_class_idx])
        is_fake = fake_prob > 0.5
        confidence = fake_prob if is_fake else 1.0 -fake_prob
        return is_fake,confidence
    
def load_demo_set(task_name):
    base = os.path.join("demo_images",task_name)
    images = []
    for label,subdir in [(0,"real"),(1,"fake")]:
        folder = os.path.join(base,subdir)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".jpg", ".jpeg",".png")):
                images.append((os.path.join(folder,fname),label))
    return images


def heatmap_creation(df):
    task_cols = []
    for col in df.columns:
        if col != "State":
            task_cols.append(col)
    percentage_display = {}
    for col in task_cols:
        percentage_display[col] = "{:.2f}%"

    return df.style.format(percentage_display).background_gradient(cmap="RdYlGn",vmin=0,vmax=100,subset=task_cols)

#def saliency_com(model,img_ten,method,class_means):
#    img_ten.requires_grad_() # remember all math operations that were done with the image that we can compute in a backward manner how every pixel was important for the result
#    model.eval() # turn of training mode
#    model.zero_grad() # deleting old calculation of derivations, such that result is not wrongly influenced
#    score = None

#    if method == "Naive Learning (1 logit)":
#        output = model(img_ten)
#        score = output

#    elif method in ["Naive Learning (2 logits)","LUCIR (Continual Learning)"]:
#        output = model(img_ten)
#        logits = output['logits']
#        pred_class = torch.argmax(logits,dim=1)
#        score = logits[0,pred_class]

#    elif method == "iCaRL (Continual Learning)":
#        if class_means is None:
#            st.error("Can't be None")
#            return None,None
        # feature extraction
#        features = model.feature_extractor(img_ten)   
        
        # L2 normalisation
#        pred_inter = (features.T / torch.norm(features.T,dim=0)).T
        
#        # calculate distance to stored class means
#        sqd = torch.cdist(class_means[:,:,0].T,pred_inter)
#        score_icarl = (-sqd).T

#        best_class = torch.argmax(score_icarl)
#        score = score_icarl[0,best_class]

#    if score is None:
#        return None,None

#    if score.dim() > 0:
#        score = score.sum()
#    score.backward() # backpropagation
#    saliency,_ = torch.max(img_ten.grad.data.abs().squeeze(),dim=0) # absolute value of gradients
#    saliency = saliency.cpu().numpy() # turned into numpy array

#    perc_max = np.percentile(saliency,99) # value under which 99% of gradient values lie
#    perc_min = saliency.min() # smallest occuring gradient value in image
#    if perc_max > perc_min:
#        saliency = np.clip((saliency-perc_min)/(perc_max-perc_min),0,1) # useful value
#    else:
#        saliency = np.zeros_like(saliency) # black 
#    heat_m = cm.hot(saliency)[...,:3] # only considering RGB
#    return heat_m,saliency


    

if __name__ == "__main__":
    st.set_page_config(page_title="Continual vs Common deepfake detection", layout="wide")
    theory_iCaRL_and_LUCIR,demo = st.tabs(["Info", "Demo"])

with theory_iCaRL_and_LUCIR:
    st.title("CONTINUAL DEEPFAKE DETECTION")
    st.title("Deepfake Detection")
    st.write("Deepfakes are images or videos that are manipulated with the help of AI (e.g. Deep Learning models) or can even be synthetically generated."
    " Common deepfake detection is made possible by training a model with a fixed amount of images/videos and known fake images/videos to recognize errors like wrong shading, unusual eye movement etc.\n " \
    "The disadvantage of common deepfake detection is that it is stale since we have a frozen model that can only detect a certain kind of deepfakes known during training. And if you wanted to teach the model to detect a new kind of deepfakes, "
    "the model will strongly forget how to detect the old kind of deepfakes.")
    st.title("Continual Learning")
    st.write( "A task can be seen as a classification problem that a model is trying to solve. " \
    "If we have now multiple tasks that a model should solve, we will see that after being trained on the newest task, it will have problems to solve an older task.\n" \
    "This happens, because the model starts to forget how to solve old tasks, since it was overwritten with the training data that is needed to solve the newest task. This phenomenon is known as 'Catastrophic Forgetting'. To prevent this effect, we are gonna use Continual Learning.\n" \
    " \n\n This means that every time we train the model, the data used for the training of the newest task is complemented by a subset of the most representative data " \
    "that was used to train the model on a previous task.")
    st.title("Continual Deepfake Detection")
    st.write("In contrast to the stationary set-up, where a large amount of deepfakes is provided all at once, " \
    "we now have the scenario of deepfakes appearing in a sequential manner. At each learning when trained on a new deepfake detection " \
    "task, a standard neural network would have problems to solve previously learned tasks due to 'Catastrophic Forgetting', which is why we will use Continual Learning, when training the model." )

    st.html("<style> body {bgcolor: #000000;}</style>")
    
    df_icarl = pd.read_csv("acc_results_from_repo/icarl_acc_matrix.csv")
    df_naive = pd.read_csv("acc_results_from_repo/naive_acc_matrix.csv")
    df_naive_l = pd.read_csv("acc_results_lucir/naive_acc_matrix.csv") 
    df_lucir = pd.read_csv("acc_results_lucir/lucir_acc_matrix.csv")
    
    st.title("Accuracy: Continual Learning Methods (iCaRL and LUCIR) vs Naive Learning")

    st.header("Comparison: iCaRL vs Naive")

    st.subheader("Naive (iCaRL code-skeleton)") # Probability of being fake
    st.dataframe(heatmap_creation(df_naive),width='stretch',hide_index=True)

    
    st.subheader("iCaRL")
    st.dataframe(heatmap_creation(df_icarl),width='stretch',hide_index=True)

    st.header("Comparison: LUCIR vs Naive") # Probability of real and fake in an array

    st.subheader("Naive (LUCIR code-skeleton)")
    st.dataframe(heatmap_creation(df_naive_l),width='stretch',hide_index=True)

   
    st.subheader("LUCIR")
    st.dataframe(heatmap_creation(df_lucir),width='stretch',hide_index=True)

   
    
    st.title("Additional Information")
    st.write("First we should clarify that we here focus on Binary Class Learning, meaning that the classification task is to differentiate between real and fake images, no matter from which fake image generator the fake image originates.")
    st.write("For this demo, we used two Continual Learning methods : Incremental Classifier and Representation Learning, which is also known as iCaRL and Learning a unified classifier incrementally via Rebalancing, which is also known as LUCIR.\n\n" \
    " To understand iCaRL, we have to explain what knowledge distillation is: \n\n " \
    " Before a model learns a new deepfake variant, we capture the knowledge of its previous state. The model is constantly penalized, while being trained on a new task, if its prediction drifts away too much from the model's prediction in the previous state. This discrepancy is also " \
    "known as the distillation loss, which we want to minimize. \n\n Additionally we should clarify what the classification loss is. It just describes the discrepancy " \
    "between the predicted label (real or fake) and the true label (real or fake). \n\n" \
    "iCaRL operates under a strict class incremental protocol, which defines the sequence of the following training and evaluation phase steps: \n\n" \
    "- Training Phase: The model first receives data for the classes of task t. During that the model has access to its exemplar memory which is a set, that contains data from the classes from task 1 to task t-1. The model improves by minimizing the knowledge distillation and classification loss.\n\n" \
    "- Evaluation Phase: The model must classify test samples after it has seen all classes from task 1 to task t.\n\n" \
    "LUCIR operates under the same strict class incremental protocol, but with two additional factors:\n\n" \
    "- Cosine normalization: If the model was trained on task i and is now trained on task i+1 with way more data in the current training data set than from task i, for which only the most representative data was chosen, we have the problem that there is a higher bias towards the current detection task. To solve this problem, " \
    "LUCIR takes the model's weights and the feature vector of the image that we wanna classify and normalizes both, such that only the orientation of the corresponding weights matter and not the amount of data, when the logits are computed.\n\n" \
    "- Feature Distillation: While the model is trained to learn a new task, the model's internal layers start to change: We compare the normalized feature representations of the current model with those of the previous model, when it gets an image from an old task as input, to force their orientations to stay aligned.\n\n " \
    "The sequence of following training and evaluation steps can be defined as:\n\n" \
    "- Training Phase: Before receiving the data for the classes of task t, the weights of the model and the output of the feature extractor, that takes input data, is normalized. During that the model has access to its exemplar memory, just as in iCaRL and it improves by minimizing the classification loss (with normalized weights and normalized feature extractor) and the feature distillation loss.\n\n" \
    "- Evaluation Phase: The model must classify test samples after it has seen all classes from task 1 to task t. ")
    
    st.header("Scenario for our demo")
    st.write("For our scenario we deal with binary classification (fake vs real) for a given model solving a given task. However what we will do, for the iCaRL method, is to add up the probabilities of being a deepfake generator class from the first task until the i-th task of five tasks, which is then the overall probability of being a fake image out of the given test set for solving task j, turning it into a binary classification problem. " \
    "This means that the accuracies that are calculated and displayed in the table show the percentage of the correctly labeled set of test images for a given task.\n\n" \
    "The rows in the table represent the state of the model, meaning on which tasks it has already been trained on. The columns represent the task on which the model is currently tested on. " 
    "In the table the accuracies on the diagonal show the performance of the model trained on the newest task t, which is tested to solve the current task t. " \
    "The accuracies under the diagonal show the performance of the model trained on the newest task t, which is tested to solve the previous tasks 1 to task t-1. " \
    "The accuracies above the diagonal show the performance of the model trained on task t, which is tested to solve task t+1 to task t_n, where n is the number of all tasks, are also known as zero-shot accuracy. " \
    "After we clarified now what the values within the tables mean, we have to mention that we have one table filled with the accuracies if the model is trained from task to task naively without considering the training data from the previous tasks, meaning we do not apply Continual Learning here. " \
    "For the other table however we used the iCaRL/LUCIR method to train the model from task to task, where we considered to include a representative subset of the training data that was used to train how to solve previous tasks. \n\n" \
    "As you will see, for the naive approach the accuracies below the diagonal will in general not look that good, which is the effect of 'Catastrophic Forgetting'. " \
    "In comparison the accuracies below the diagonal for the iCaRL/LUCIR table will look pretty good, showing the benefits of Continual Learning. " \
    "The accuracies above the diagonal are low for both approaches due to the fact that the model has not yet seen the data to solve these tasks. \n\n" \
    "Additionally it should be mentioned that LUCIR does perform better in general as well as for our demo, which can be seen through the average accuracy per state due to the additional measures that LUCIR takes.")

    st.title("Citation")
    st.write("The code used for the training originates from these authors and their corresponding paper: ")
    st.markdown("""
    > **Li, C., Huang, Z., Paudel, D. P., Wang, Y., Shahbazi, M., & Van Gool, L. (2023).**  
    > *A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials.*  
    > In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).
    """)
    bibtex_citation = """@inproceedings{li2022continual,
      title={A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials},
      author={Li, Chuqiao and Huang, Zhiwu and Paudel, Danda Pani and Wang, Yabin and Shahbazi, Mohamad and Van Gool, Luc},
      booktitle={Winter Conference on Applications of Computer Vision (WACV)},
      year={2023}
    }"""
    
    st.code(bibtex_citation, language="bibtex")


    # it is normal that naive model has higher confidence especially for d2 tested on task t1 due
    # to the fact that the softmax applied to the eucledian distance yields lower probabilities but more stable for detectors in a later state
with demo:
    av_acc = {"Naive Learning":{1:100.0, 2:100.0,3:66.7,4:75.0,5:58.0}, "iCaRL":{1:100.0,2:100.0,3:100.0,4:95.0, 5:80.0}}
    av_acc_l = {"Naive Learning":{1:100.0, 2:95.0,3:66.7,4:60.0,5:62.0}, "LUCIR":{1:100.0,2:100.0,3:100.0,4:97.5, 5:88.0}}
    st.header("Single Picture Evaluation")
    st.info("**What is this demo about?**:  You can test how well a deepfake detector can recognize fake images. Compare a neural network detector trained in a standard way with a neural network detector trained via Continual Learning. ")
    col_m, col_s, col_t = st.columns(3)

    with col_m:
        st.markdown("#### 1. Choose method")
        method = st.radio("Training method used:",["Naive Learning (iCaRL code-skeleton)", "Naive Learning (LUCIR code-skeleton)","iCaRL (Continual Learning)", "LUCIR (Continual Learning)"],help=" 'Naive' means that the model is trained in such a way, such that it forgets how to differentiate between real and fake for an old task, i.e. to differentiate between the fake images of a certain generator and real images.\n\n" \
        "'Continual' means that the training set is updated from task to task, such that this 'Forgetting' is mitigated. \n\n Furthermore, we have two 'Naive modes', because when setting the naive flag in the iCaRL code-skeleton," \
        "the model is initialized in such a way that it has only one output neuron putting out the logit/probability of being a fake image, whereas in the LUCIR code-skeleton the model is initialized in such a way that it has two output neurons putting out" \
        "two logits, where the 0-th value is the logit/probability of an image being real and the 1-st value is the logit/probability of an image being fake. Additionally, when" \
        "initializing the model in the LUCIR code-skeleton, the features and the weights of the model are normalized before the logits are computed, which does not happen in the iCaRL code-skeleton. ")

    with col_s:
        st.markdown("#### 2. Choose detector")
        sel_state_idx = st.selectbox("State:",options=list(task_dict.keys()),format_func=lambda i: f"Detector {i} (trained to solve task {"1" if i == 1 else "1 to " f"{i}" })",key="Select iCaRL or naive state")
        naive_avg_l = av_acc_l["Naive Learning"][sel_state_idx]
        naive_avg = av_acc["Naive Learning"][sel_state_idx]
        icarl_avg = av_acc["iCaRL"][sel_state_idx]
        lucir_avg = av_acc_l["LUCIR"][sel_state_idx]
        if method == "Naive Learning (iCaRL code-skeleton)":
            cur_avg = naive_avg
        elif method == "Naive Learning (LUCIR code-skeleton)":
            cur_avg = naive_avg_l
        elif method == "iCaRL (Continual Learning)":
            cur_avg = icarl_avg
        else:
            cur_avg = lucir_avg
        st.caption(f" Average accuracy of detector for state {sel_state_idx} = {cur_avg} %",help="The average accuracy for each state (or the i-th row) is calculated by taking the accuracies from the testings of task 1 to task t (or column 1 to column t), where t is the number of the last task that the model has been trained on and divide them by the number of tasks t.")

    with col_t:
        st.markdown("#### 3. Choose test domain")
        sel_task_idx = st.selectbox(f"Test set:",options=list(task_dict.keys()),format_func=lambda i: f"Choose validation set to test how detector solves task {i}",key="Select test set (iCaRL or naive)")

    st.divider()

    # Choice of image
    task_name = task_dict[sel_task_idx]
    demo_imgs = load_demo_set(task_name)

    if not demo_imgs:
        st.warning(f"No images found in folder for task {task_name}")

    else:
        def file_name_gt(index):
            path_img,label = demo_imgs[index] 
            file_name = os.path.basename(path_img)
            gr_tr = "Real" if label == 0 else "Fake"
            return f"Image {index+1} : {file_name} (Ground truth : {gr_tr})"

        sel_img_idx = st.selectbox("Choose a validation image",options=range(len(demo_imgs)),format_func=file_name_gt,key="Selected image_iC")
        sel_img_path,tr_la = demo_imgs[sel_img_idx]
        true_label = "Real" if tr_la == 0 else "Fake" 

        # loading model and doing inference
        img_r = Image.open(sel_img_path).convert("RGB")
        img_ten = Transform(img_r).unsqueeze(0)
        is_fake, confidence = None,None

        if method == "Naive Learning (iCaRL code-skeleton)":
            model = load_naive_model(sel_state_idx)
            if model is not None:
                is_fake,confidence = naive_inference(model,img_ten)
        elif method == "iCaRL (Continual Learning)":
            model,class_means = load_model_and_means(sel_state_idx,total_tasks=5)
            if model is not None and class_means is not None:
                is_fake,confidence = icarl_inference(model,class_means,img_ten)
        elif method =="Naive Learning (LUCIR code-skeleton)":
            model = load_naive_model_luc(sel_state_idx)
            if model is not None:
                is_fake,confidence = lucir_inference(model,img_ten)
        else:
            model = load_lucir_model(sel_state_idx)
            if model is not None:
                is_fake,confidence = lucir_inference(model,img_ten)
        
      
        # visualization of result
        col_img,col_res = st.columns([1,1])

        with col_img:
            st.subheader("Chosen Picture")
            st.image(img_r,width='stretch', caption=f"File : {os.path.basename(sel_img_path)}")

        with col_res:
            st.subheader("Prediction")

            if is_fake is None:
                st.error("Model or class means could not be loaded")

            else:
                pred_label = "Fake" if is_fake else "Real"
                correct = int(is_fake)== tr_la

                met_col_1, met_col_2 = st.columns(2)

                with met_col_1:
                    st.metric("Ground Truth",true_label)

                with met_col_2:
                    st.metric("Prediction",pred_label)

                st.metric(f"Confidence/Probability that image is {pred_label} ",f"{confidence*100:.2f}")

                if correct:
                    st.success(f"The model recognizes the image correctly as {pred_label}")
                else:
                    st.error(f"The model recognizes the image wrongly as {pred_label}, although it was {true_label} ")

                # saliency map

               # st.divider()
               # st.subheader("Explainability of Model Decision")
               # if st.button("Build Saliency Map", help= "The Saliency Map shows which pixels were relevant for the prediction of the model."):
               #     with st.spinner("Gradients are calculated"):
               #         img_ten_sal = Transform(img_r).unsqueeze(0)
               #         c_m = class_means if method == "iCaRL (Continual Learning)" else None
               #         if model is not None:
               #             heatmap , saliency = saliency_com(model,img_ten_sal,method,c_m)
               #         else:
               #             st.error("Model what are you doing")
               #         # undoing normalization such that heatmap can be layed on image
               #         mean_inv = [-0.485/0.229,-0.456/0.224,-0.406/0.225]
               #         std_inv = [1/0.229,1/0.224,1/0.225]

               #         ten_clone = img_ten.squeeze().clone()

               #         for ten,mean,std_ in zip(ten_clone,mean_inv,std_inv): # get original RGB values before normalization
               #             ten.sub_(mean).div_(std_)

               #         # streamlit needs image as [height,width,channel] but pytorch stores images as [channel,height,width] 

               #         img = ten_clone.permute(1,2,0).numpy()
               #         img = np.clip(img,0,1)
               #         over_l = 0.5 * img + 0.5*heatmap

               #         sali_col_1,sali_col_2 = st.columns(2)
               #         with sali_col_1: # without picture only regions
               #             st.image(heatmap,caption="Saliency Heatmap",width='stretch')

               #         with sali_col_2: # with picture
               #             st.image(over_l,caption="Overlay on image",width='stretch')
               #         st.caption("Yellow/White regions show most important spots for model evaluation.") 


    
  
                    
                    
                       


    
    