import streamlit as st
import torch
from CNND_model import Model_CNND
import os
import torchvision.transforms as transforms
from PIL import Image
from argparse import Namespace
import pandas as pd
import gc
import sys
import models_lucir.modified_resnet
from models_lucir.modified_resnet import resnet50

sys.modules['models'] = sys.modules.get('models_lucir')
sys.modules['models.modified_resnet'] = models_lucir.modified_resnet

#torch.set_num_threads(4)

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
def load_naive_model_luc(state_idx):
    file_name = f"naive_model_state_{state_idx}_after_t{state_idx-1}.pt"
    path = os.path.join(checkpoint_naive_dir_l,file_name)
    if not os.path.exists(path):
            return None

    checkpoint = torch.load(path,map_location=torch.device('cpu'),weights_only=False)
    if isinstance(checkpoint,torch.nn.Module):
        model = checkpoint
    else:
        class_num_cur = 2*state_idx
        model = resnet50(num_classes=class_num_cur,pretrained=False)
        state_dict = checkpoint.get('state_dict',checkpoint) if isinstance(checkpoint,dict) else checkpoint.state_dict()
        model.load_state_dict(state_dict)

    model.eval()
    return model

@st.cache_resource(max_entries=1)
def load_lucir_model(state_idx):
    file_name = f"model_state_{state_idx}_after_t{state_idx-1}.pt"
    path = os.path.join(checkpoint_dir_l,file_name)
    if not os.path.exists(path):
            return None
    checkpoint = torch.load(path,map_location=torch.device('cpu'),weights_only=False)
    if isinstance(checkpoint,torch.nn.Module):
        model = checkpoint
    else:
        class_num_cur = 2*state_idx
        model = resnet50(num_classes=class_num_cur,pretrained=False)
        state_dict = checkpoint.get('state_dict',checkpoint) if isinstance(checkpoint,dict) else checkpoint.state_dict()
        model.load_state_dict(state_dict)
    model.eval()
    return model
    

@st.cache_resource(max_entries=1)
def load_model_and_means(state_idx,total_tasks=5):
# Model_State laden

   
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
    

def naive_inference(model,img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        pred_prob = output.item()
        is_fake = pred_prob > 0.5
        confidence = pred_prob if is_fake else (1-pred_prob)
        return is_fake,confidence,pred_prob

def icarl_inference(model,class_means,img_tensor):
    if class_means is None:
        return None,None,None

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

        return is_fake,confidence,fake_prob

def lucir_inference(model,img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        logits = output['logits']
        probs = torch.softmax(logits,dim=1)
        fake_class_idx = [i for i in range(logits.shape[1]) if i%2 == 1]
        fake_prob = sum([probs[0,idx].item() for idx in fake_class_idx])
        is_fake = fake_prob > 0.5
        confidence = fake_prob if is_fake else 1.0 -fake_prob
        return is_fake,confidence,fake_prob
    
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

def load_demo_set_large(task_name):
    base = os.path.join("demo_images_larger",task_name)
    images = []
    for label,subdir in [(0,"real"),(1,"fake")]:
        folder = os.path.join(base,subdir)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".jpg", ".jpeg",".png")):
                images.append((os.path.join(folder,fname),label))
    return images
    
#def evaluate_state_on_set_with_both(model_naive,model_icarl,class_means,image_label_pairs):
 #   naive_cor = 0 if model_naive is not None else None
  #  icarl_cor = (0 if model_icarl is not None and class_means is not None else None)
   # n = len(image_label_pairs)
    #if n == 0:
     #   return None,None
    #for path,true_label in image_label_pairs:
    #    with Image.open(path) as r_img:
    #        img = r_img.convert("RGB")
    #        img_tensor = Transform(img).unsqueeze(0)
    #    if model_naive is not None:
    #        naive_is_fake,_,_ = naive_inference(model_naive,img_tensor)
    #        naive_cor += int(int(naive_is_fake)==true_label)

     #   if model_icarl is not None and class_means is not None:
      #      icarl_is_fake,_,_ = icarl_inference(model_icarl,class_means,img_tensor)
       #     if icarl_is_fake is not None:
        #        icarl_cor += int(int(icarl_is_fake)==true_label)
        
    #naive_acc = naive_cor/n if naive_cor is not None else None
    #icarl_acc = icarl_cor /n if icarl_cor is not None else None
    #return naive_acc,icarl_acc

def heatmap_creation(df):
    task_cols = []
    for col in df.columns:
        if col != "State":
            task_cols.append(col)
    percentage_display = {}
    for col in task_cols:
        percentage_display[col] = "{:.2f}%"

    return df.style.format(percentage_display).background_gradient(cmap="RdYlGn",vmin=0,vmax=100,subset=task_cols)
    

if __name__ == "__main__":
    st.set_page_config(page_title="Continual vs Common deepfake detection", layout="wide")
    theory_iCaRL_and_LUCIR,demo_10_img_iCaRL,demo_10_img_LUCIR = st.tabs(["Matrix Eval (iCaRL)", "Interactive Selection (iCaRL)", "Interactive Selection (LUCIR)"])

with theory_iCaRL_and_LUCIR:
    st.title("CONTINUAL DEEPFAKE DETECTION")
    st.title("Deepfake Detection")
    st.write("Before we explain what deepfake detection is, let us shortly explain what deepfakes are: \n\n Deepfakes are images or videos that are manipulated with the help of AI(e.g. Deep Learning models) or can even be synthetically generated"
             ".\n They pose a major threat, when they are used in a medial context to influence people or if they are used to create explicit content of a person without their consent. To mitigate these problems, we need Deepfake Detection."
             " How could this be realized?: \n\n" \
    " Common deepfake detection is made possible by training a model with a fixed amount of images/videos and known fake images/videos to recognize errors like wrong shading, unusual eye movement etc.\n " \
    "The disadvantage of common deepfake detection is that it is stale since we have a frozen model that can only detect a certain kind of deepfakes known during training. And if you wanted to teach the model to detect a new kind of deepfakes, "
    "the model will strongly forget how to detect the old kind of deepfakes."
    " This means that if you want to be able to detect multiple kind of deepfakes, this will lead to the need of retraining the model from zero again and again.\n" \
    "This is where Continual Learning will come into play.")
    st.title("Continual Learning")
    st.write( "Before explaining what Continual Learning is, we want to explain what a task is. A task can be seen as a classification problem that a model is trying to solve. " \
    "A task for example would be: A model should classify, given a test picture, if there is a dog on the picture or a cat. Another example would be to classify between a car and a motorcycle. " \
    "If we have now multiple tasks that a model should solve, we will see that after being trained on the newest task, it will have problems to solve an older task.\n\n" \
    "Why?: Because the model starts to forget how to solve old tasks, since it was overwritten with the training data that is needed to solve the newest task.\n" \
    "To mitigate this, so called, 'Catastrophic forgetting', we use Continual Learning. But what does that mean?: \n\n It simply means that every time we train the model, the used training data for the training of the newest task is complemented by a subset of the training data " \
    "that was used to train the model on a previous task. This subset is selected in such a way that it is representative for the old classes." \
    " By this technique the model is able to remember the properties of the old classes such that it can use this newly won memory to solve older tasks in a much better way.")
    st.title("Continual Deepfake Detection")
    st.write("Now we want to combine Continual Learning and Deepfake Detection. This means that in contrast to the stationary set-up, where a large amount of deepfakes is provided all at once," \
    "we now have the scenario of deepfakes appearing time by time in a sequential manner. At each learning when trained on a new deepfake detection " \
    "task, a standard neural network would have problems to solve previously learned tasks due to the 'Catastrophic Forgetting' that we already talked about.\n\n" \
    "Continual Learning gives us the possibility to mitigate this problem by updating the model dynamically with new deepfake detection tasks without forgetting how to solve the old ones.")
    
    #st.image(r"C:\Users\User\Desktop\CySec_Project_CDD\demo_streamlit\demo_results_larger\Screenshot 2026-08-13 234025.png")
    #st.image(r"C:\Users\User\Desktop\CySec_Project_CDD\demo_streamlit\demo_results_larger\Screenshot 2026-08-13 234110.png")
    df_icarl = pd.read_csv("acc_results_from_repo/icarl_acc_matrix.csv")
    df_naive = pd.read_csv("acc_results_from_repo/naive_acc_matrix.csv")
    df_lucir = pd.read_csv("acc_results_lucir/lucir_acc_matrix.csv")
    df_naive_lu = pd.read_csv("acc_results_lucir/naive_acc_matrix.csv")

    st.title("Accuracy: Continual Learning (iCaRL and LUCIR) vs Naive Learning")

    st.header("Continual Learning: Method 1 = iCaRL")
    st.subheader("iCaRL")
    st.dataframe(heatmap_creation(df_icarl),width='stretch',hide_index=True)

    st.subheader("Naive")
    st.dataframe(heatmap_creation(df_naive),width='stretch',hide_index=True)

    st.header("Continual Learning: Method 2 = LUCIR")
    st.subheader("LUCIR")
    st.dataframe(heatmap_creation(df_lucir),width='stretch',hide_index=True)

    st.subheader("Naive")
    st.dataframe(heatmap_creation(df_naive_lu),width='stretch',hide_index=True)
    
    st.title("Additional Information")
    st.write("First we should clarify that we here focus on Binary Class Learning, meaning that the classification task is to differentiate between real and fake images, no matter from which fake image generator the fake image originates.")
    st.write("For this demo, we used two Continual Learning methods : Incremental Classifier and Representation Learning, which is also known as iCaRL and Learning a unified classifier incrementally via Rebalancing, which is also known as iCaRL. But how do these two methods work?:\n\n" \
    " Before explaining what iCaRL concretely does, we first have to explain, what knowledge distillation is: \n\n " \
    " Before a model learns a new deepfake variant, we capture the knowledge of its previous state. The model is constantly penalized, while being trained on a new task, if its prediction drifts away too much from the model's prediction in the previous state. This discrepancy is also " \
    "known as the distillation loss, which we want to minimize. \n\n Additionally we should clarify what the classification loss is. It just describes the discrepancy " \
    "between the predicted label (real or fake) and the true label (real or fake) \n\n Now that we know what knowledge distillation and what classification loss is, we can explain what iCaRL does: \n\n" \
    "iCaRL operates under a strict class incremental protocol, which defines the sequence of the following training and evaluation phase steps: \n\n" \
    "- Training Phase: The model first receives data for the classes of task t. During that the model has access to its exemplar memory which is a set, that contains data from the classes from task 1 to task t-1. The model improves by minimizing the knowledge distillation and classification loss.\n\n" \
    "- Evaluation Phase: The model must classify test samples after it has seen all classes from task 1 to task t.\n\n" \
    "LUCIR operates under the same strict class incremental protocol, but with two additional factors:\n\n" \
    "- Cosine normalization: If the model was trained on task i and is now trained on task i+1 with way more data in the current training data set than from task i, for which only the most representative data was chosen, we have the problem that there is a higher bias towards the current detection task. To solve this problem" \
    "LUCIR takes the model's weights and the feature vector of the image that we wanna classify and normalizes both, such that only the orientation of the corresponding weights matter and not the amount of data.\n\n" \
    "- Feature Distillation: While the model is trained to learn a new task, the model's internal layers start to change: We compare the normalized feature representations of the current model with those of the previous model to force their orientations to stay aligned. " \
    "such that the sequence of following training and evaluation steps can be defined:\n\n" \
    "- Training Phase: Before receiving the data for the classes of task t, the weights of the model and the output of the feature extractor, that takes input data, is normalized. During that the model has access to its exemplar memory, just as in iCaRL and it improves by minimizing the classification loss (with normalized weights and normalized feature extractor) and the feature distillation loss.\n\n" \
    "- Evaluation Phase: The model must classify test samples after it has seen all classes from task 1 to task t. ")
    
    st.header("Scenario for our demo")
    st.write("For our scenario we deal, as already mentioned, with binary classification (fake vs real) even if there are seven deepfake detection tasks given. However what we will then just do is to add up the probabilities of being a deepfake generator class from the first task until the i-th task of seven tasks, which is then the overall probability of being a fake image. " \
    "This means that the accuracies that are calculated and displayed in the table show, how many times we correctly labeled a set of test images as fake/real for a given task\n\n" \
    "The rows in the table represent the state of the model, meaning on which tasks it has already been trained on. The columns represent the task on which the model is currently tested on. " 
    "In the table the accuracies on the diagonal show the performance of the model trained on the newest task t, which is tested to solve the current task t. " \
    "The accuracies under the diagonal show the performance of the model trained on the newest task t, which is tested to solve the previous tasks 1 to task t-1." \
    "The accuracies above the diagonal show the performance of the model trained on task t, which is tested to solve task t+1 to task t_n, where n is the number of all tasks, also known as zero-shot accuracy. " \
    "After we clarified now what the values within the tables mean, we have to mention that we have one table filled with the accuracies if the model is trained from task to task naively without considering the training data from the previous tasks, meaning we do not apply Continual Learning here. " \
    "For the other table however we used the iCaRL/LUCIR method to train the model from task to task, where we considered to include a representative subset of the training data that was used to train how to solve previous tasks. \n\n" \
    "As you will see, for the naive approach the accuracies below the diagonal will in general not look that good, which is the effect of 'Catastrophic Forgetting, that we already talked about. " \
    "In comparison the accuracies below the diagonal for the iCaRL/LUCIR table will look pretty good, showing the benefits of Continual Learning. " \
    "The accuracies above the diagonal are low for both approaches due to the fact that the model has not yet seen the data to solve these tasks. \n\n" \
    "Additionally it should be mentioned that LUCIR does perform better in general as well as for our demo, which can be seen through the average accuracy per state due to the additional measures that LUCIR takes, which was explained above.")

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
with demo_10_img_iCaRL:
    av_acc = {"Naive Learning":{1:100.0, 2:100.0,3:66.7,4:75.0,5:58.0}, "iCaRL":{1:100.0,2:100.0,3:100.0,4:95.0, 5:80.0}}
    st.header("Single Picture Evaluation")
    st.info("**What is this demo about?**:  You can test how well a deepfake detector can recognize fake images. Compare a neural network detector trained in a standard way with a neural network detector trained via Continual Learning. ")
    col_m, col_s, col_t = st.columns(3)

    with col_m:
        st.markdown("#### 1. Choose method")
        method = st.radio("Training method used:",["Naive Learning","iCaRL(Continual Learning)"],help=" 'Naive' means that the model is trained in such a way, such that it forgets how to differentiate between real and fake for an old task, i.e. to differentiate between the fake images of a certain generator and real images.\n\n" \
        "'Continual' means that the training set is updated from task to task, such that this 'Forgetting is mitigated.")

    with col_s:
        st.markdown("#### 2. Choose detector")
        sel_state_idx = st.selectbox("State:",options=list(task_dict.keys()),format_func=lambda i: f"Detector {i} (trained to solve task {"1" if i == 1 else "1 to " f"{i}" })",key="Select iCaRL or naive state")
        naive_avg = av_acc["Naive Learning"][sel_state_idx]
        icarl_avg = av_acc["iCaRL"][sel_state_idx]
        cur_avg = naive_avg if method == "Naive Learning" else icarl_avg
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
        is_fake, confidence, prob = None,None,None

        if method == "Naive Learning":
            model = load_naive_model(sel_state_idx)
            if model is not None:
                is_fake,confidence,prob = naive_inference(model,img_ten)
        else:
            model,class_means = load_model_and_means(sel_state_idx,total_tasks=5)
            if model is not None and class_means is not None:
                is_fake,confidence,prob = icarl_inference(model,class_means,img_ten)

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

with demo_10_img_LUCIR:
    av_acc_l = {"Naive Learning":{1:100.0, 2:95.0,3:66.7,4:60.0,5:62.0}, "LUCIR":{1:100.0,2:100.0,3:100.0,4:97.5, 5:88.0}}
    st.header("Single Picture Evaluation")
    st.info("**What is this demo about?**:  You can test how well a deepfake detector can recognize fake images. Compare a neural network detector trained in a standard way with a neural network detector trained via Continual Learning. ")
    col_m_l, col_s_l, col_t_l = st.columns(3)
    
    with col_m_l:
        st.markdown("#### 1. Choose method")
        method = st.radio("Training method used:",["Naive Learning","LUCIR(Learning a Unified Classifier Incrementally via Rebalancing)"],help=" 'Naive' means that the model is trained in such a way, such that it forgets how to differentiate between real and fake for an old task, i.e. to differentiate between the fake images of a certain generator and real images.\n\n" \
        "'Continual' means that the training set is updated from task to task, such that this 'Forgetting is mitigated.")
    
    with col_s_l:
        st.markdown("#### 2. Choose detector")
        sel_state_idx = st.selectbox("State:",options=list(task_dict.keys()),format_func=lambda i: f"Detector {i} (trained to solve task {"1" if i == 1 else "1 to " f"{i}" })",key="Select LUCIR state or naive")
        naive_avg = av_acc_l["Naive Learning"][sel_state_idx]
        lucir_avg = av_acc_l["LUCIR"][sel_state_idx]
        cur_avg = naive_avg if method == "Naive Learning" else lucir_avg
        st.caption(f" Average accuracy of detector for state {sel_state_idx} = {cur_avg} %",help="The average accuracy for each state (or the i-th row) is calculated by taking the accuracies from the testings of task 1 to task t (or column 1 to column t), where t is the number of the last task that the model has been trained on and divide them by the number of tasks t.")
    
    with col_t_l:
        st.markdown("#### 3. Choose test domain")
        sel_task_idx = st.selectbox(f"Test set:",options=list(task_dict.keys()),format_func=lambda i: f"Choose validation set to test how detector solves task {i}",key= "Select test set(LUCIR or naive)")
    
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
    
        sel_img_idx = st.selectbox("Choose a validation image",options=range(len(demo_imgs)),format_func=file_name_gt,key="Selected img (LUCIR or naive)")
        sel_img_path,tr_la = demo_imgs[sel_img_idx]
        true_label = "Real" if tr_la == 0 else "Fake" 
    
        # loading model and doing inference
        img_r = Image.open(sel_img_path).convert("RGB")
        img_ten = Transform(img_r).unsqueeze(0)
        is_fake, confidence, prob = None,None,None
    
        if method == "Naive Learning":
            model = load_naive_model_luc(sel_state_idx)
            if model is not None:
                is_fake,confidence,prob = lucir_inference(model,img_ten)
        else:
            model = load_lucir_model(sel_state_idx)
            if model is not None:
                is_fake,confidence,prob = lucir_inference(model,img_ten)
    
        # visualization of result
        col_img_l,col_res_l = st.columns([1,1])
    
        with col_img_l:
            st.subheader("Chosen Picture")
            st.image(img_r,width='stretch', caption=f"File : {os.path.basename(sel_img_path)}")
    
        with col_res_l:
            st.subheader("Prediction")
    
            if is_fake is None:
                st.error("Model or class means could not be loaded")
    
            else:
                pred_label = "Fake" if is_fake else "Real"
                correct = int(is_fake)== tr_la
    
                met_col_1_l, met_col_2_l = st.columns(2)
    
                with met_col_1_l:
                    st.metric("Ground Truth",true_label)
    
                with met_col_2_l:
                    st.metric("Prediction",pred_label)
    
                st.metric(f"Confidence/Probability that image is {pred_label} ",f"{confidence*100:.2f}")
    
                if correct:
                    st.success(f"The model recognizes the image correctly as {pred_label}")
                else:
                    st.error(f"The model recognizes the image wrongly as {pred_label}, although it was {true_label} ")
                    
                    
                       


    
    