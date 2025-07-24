### 📅 Day 2 Progress – Adversarial ML Red Team Toolkit

#### ✅ **Objectives for Day 2**

*   Generate adversarial examples using FGSM
    
*   Visualize original vs adversarial images
    
*   Evaluate model accuracy drop after adversarial perturbation
    
*   Integrate everything into the CLI workflow
    

### 🧠 Key Concepts Covered

*   **FGSM (Fast Gradient Sign Method):** A popular method to create adversarial examples by tweaking the input image slightly in the direction that maximally increases the loss.
    
*   **Adversarial Example:** A perturbed version of a valid input that causes the model to make incorrect predictions.
    
*   **Model Robustness Evaluation:** Measuring how much accuracy drops when adversarial examples are introduced.
    
*   **Matplotlib Visualization:** To visually compare clean vs adversarial images.
    
*   **Impact Analysis:** Quantifying the effect of adversarial examples on model performance.
    
*   **Modular Design:** Separating code into fgsm.py, evaluate\_impact.py, and using main.py as the orchestrator.
    

### 🛠️ Code and Module Summary

FilePurposesrc/attacks/fgsm.pyGenerates adversarial images using FGSMsrc/evaluation/evaluate\_impact.pyCalculates model accuracy on clean and adversarial samplesmain.pyLoads model, dataset, runs attack, visualizes outputs, and prints impactutils/visualize.py(Optional) Shows side-by-side comparison of clean and adversarial images using matplotlib

### 🔁 What Happens When You Run python main.py

1.  **Load Dataset**:
    
    *   Downloads and loads MNIST test dataset.
        
2.  **Load Model**:
    
    *   Loads a pretrained or defined model for MNIST digit classification.
        
3.  **Generate and Visualize**:
    
    *   Displays a randomly selected MNIST image.
        
    *   Generates its adversarial version using FGSM.
        
    *   Shows both clean and perturbed images via matplotlib.
        
4.  **Make Predictions**:
    
    *   Displays the model's prediction on both original and adversarial image.
        
5.  **Impact Evaluation**:
    
    *   Runs the full test dataset through the model.
        
    *   Prints the clean accuracy, adversarial accuracy, and accuracy drop.
        

### 📈 Example Output (from terminal)

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   $ python main.py  INFO:root:Loading MNIST dataset...  Original Label: 7, Prediction: 6  Adversarial Prediction: 6  Accuracy on clean data: 10.43%  Accuracy on adversarial data: 7.60%  Accuracy drop: 2.83%   `

### 🧾 Tasks Completed

*   Built FGSM attack and generated perturbed images.
    
*   Visualized original and adversarial images.
    
*   Wrote evaluate\_impact.py to measure accuracy drop.
    
*   Integrated everything with main.py.
    
*   Tested and debugged visualization and evaluation.
    
*   Verified CLI usage and model output.
    

### 🔍 Learnings

*   Even slight perturbations can fool deep learning models.
    
*   Visualization is key to intuitively understand adversarial attacks.
    
*   Measuring model robustness quantitatively (accuracy drop) is essential in red teaming.