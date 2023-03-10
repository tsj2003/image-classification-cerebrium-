**Deploy Classification Neural Network on Serverless GPU platform of Banana Dev**


You have to develop a deployment pipeline to deploy Machine Learning Model on Banana Dev [https://www.banana.dev].
You have to use Gihub for the codebase and Gihub Actions to run and develop the pipeline.

**Assignment Protocols**
- You will be given 6 hours at max to complete this assessment
- You can only use Python as a programming Language
- Once you start you have to complete this assessment in those 6 hours, it cannot be paused and resumed later
- You cannot take help from any other person
    - But you can use google to search for references
- Record a 5-10 mins of code walkthrough of the work you have done. You can use Loom Platform (https://www.loom.com) to record the video.
    - A live demo of each of the features mentioned below:
        - Github Action Pipeline
        - Banana.dev platform deployment page
        - Other scripts as required in "Deliverable" section below
    - Code overview of each of those features:
        - Why did you implement it that way?
        - Is there any way you would improve it?
    - Explain what tests you have developed and how
    - Explain what parts of the assessment are completed and what is missing?
    - Make sure to submit the screen recording link in the submission after you are done recording
    - Please note that the free plan on Loom only allows for videos up to 5 minutes in length. As such, you may need to record two separate 5-minute videos.

**Model Details:**
- Model is designed to perform classification on an input Image.
- PyTorch Implementation of model is present in pytorch_model.py, and weights can be downloaded from this link: https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0
- The model is trained on ImageNet Dataset [https://www.image-net.org]
- The input to the model is an image of size 224x224, and the output is the array with probabilities for each class.
- The length of the output array is equal to the number of classes [1000] in the ImageNet dataset.

Model is trained on images with specific pre-processing steps, e.g. you need to do the following pre-processings on the image before passing it to the model. A function (preprocess_numpy) is implemented in the model class which performs the necessary pre-processing on the image, and at the end of pytorch_model.py you can see how to use the model on an image file to get inference.
- Convert to RGB format if needed. The model accepts the image in RGB format, not in BGR. Code will never throw errors for this so keep an eye on the library you use to load image.
- Resize to 224x244 (use bilinear interpolation)
- Divide by 255
- Normalize using mean values for each channel [RGB][0.485, 0.456, 0.406] and standard deviation values for each channel [RGB] [0.229, 0.224, 0.225]
    - subtract mean and divide standard deviation per channel

**Deliverable**
- convert_to_onnx.py | codebase to convert the PyTorch Model to the ONNX model
- test_onnx.py | codebase to test the converted onnx model on CPU
    - There are two images in this repo, your file should run onnx model on these images and verify if the model outputs the correct class id and class name
        - n01440764_tench belongs to class id 0
        - n01667114_mud_turtle belongs to class id 35
    - The test should report failure in case the outputs are not correct and the pipeline should report the failed test cases    
- model.py with the following classes/functionalities, make their separate classes:
    - Onnx Model loading and prediction call
    - Pre-processing of the Image [Sample code provided in pytorch_model.py]
- Things needed to deploy the code to the banana dev
- test_server.py | codebase to make a call to the model deployed on the banana dev
    - This should accept the path of the image and return/print the name of the class the image belongs to
    - And also accept a flag to run preset custom tests, where you make calls to the banana deployed models using the two images and verify results like you are expected to do in test_onnx.py
    - Also report the time one banana dev call takes
- GitHub Actions pipeline that can do the following:
    - Pre-Deployment Pipeline:
        - Should be run manually through the Actions tab on GitHub
        - Build the docker image
- Readme File | which has steps to run/use all the deliverables with proper details, such that a person who has no prior information about this repo can understand and run this easily with no blockers.
- Hint: You need to use Deploy From Github option, and make a fork of their public template.
    - Follow the steps mentioned in the banana dev documentation

**Evaluation Criteria**
 - *Python* best practices
 - Completeness: Did you include all features?
 - Correctness: Does the solution (all deliverables) work in sensible, thought-out ways?
 - Maintainability: Is the code written in a clean, maintainable way?
 - Testing: Is the solution adequately tested?
 - Documentation: Is the codebase well-documented and has proper steps to run any of the deliverables?

 
**Things which are very important and will be considered during evaluation**
- Your test_server.py should be properly implemented, we will use that to test your final deployment.
- Don't deploy PyTorch Model, you need to convert the PyTorch Model to ONNX first and use that in the deployment.
- Code Formatting and Documentation.
- Proper use of Git and Github Actions.
- Meaningful and good commits, we will monitor commit history.
- Don't delete the other branch you make all the commits to.
    - Your flow can be you make changes in a new branch made from the main.
    - And then at the end when you are done with changes you merge the new branch using proper protocols in the main for banana dev deployment

**Extra Points:**
- Make pre-processing steps part of Onnx File [<name_of_model>.onnx file], which needs to be done during onnx conversion, instead of implementing them in the code inside app.py.
- Create a GitHub Actions pipeline such that if Docker Image successfully builds and test cases pass it automatically merges the PR into the main.

**Note:**
- You would need to enter the debit/credit card details on the banana.dev platform but you will not be charged.
- They would charge 1$ for testing of the card, which gets refunded automatically by banana dev.
- You get 1 hour of Free GPU and unlimited CPU calls.
- Ideally, you should be charged zero USD for this assignment unless there is some bug on the banana platform or the codebase you develop.
- In case, some mishap occurs because of banana dev or your mistake and you are charged an extra amount MTailor is not accountable for that.
