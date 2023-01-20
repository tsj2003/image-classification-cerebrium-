**Deploy Classification Neural Network on Serverless GPU platform of Banana Dev**


You have to develop a deployment pipeline to deploy Machine Learning Model on Banana Dev [https://www.banana.dev].
You have to use Gihub for codebase and Gihub Actions to run and develop the pipeline.

**Assignment Protocols**
- You will be given 5 hours at max to complete this assessment
- You can only use Python as programming Language
- Once you start you have to complete this assessment in those 5 hours, it cannot be paused and resumed later
- You cannot take help from any other person
    - But you can use google to search for references
- Record videos for whole duration of doing the assessment:
    - Make sure you start screen recording before starting the assessment and submit the screen recording link after you are done with the assessment
    - Make sure you record your self while doing the assignment, you can do this either using laptop camera or you can use mobile phone to make your video.
    - The two recordings should be easily downloadable, and should require no authentication

**Model Details:**
- Model is deisgned to perform classification on an input Image.
- Pytorch Implementation of model is present in pytorch_model.py, and weights can be downloaded from this link: https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0
- The model is trained on ImageNet Dataset [https://www.image-net.org]
- The input to the model is an image of size 224x224, and output is the array with probabilities for each class.
- Length of the output array is equal to number of classes [1000] in the ImageNet dataset.

Model is trained on images with specific pre-processing steps, e.g. you need to do the following pre-processings on the image before passing it to the model:
- Convert to RGB format if needed. Model accepts the image in RGB format not in BGR. Code will never throw error for this so keep an eye on library you use to load image.
- Resize to 224x244 (use bilinear interpolation)
- Divide by 255
- Nomalize using mean values for each channel [RGB][0.485, 0.456, 0.406] and standard deviation values for each channel [RGB] [0.229, 0.224, 0.225]
    - subtract mean and divide standard deviation per channel

**Deliverable**
- convert_to_onnx.py | codebase to convert the PyTorch Model to the ONNX model
- test_onnx.py | codebase to test the converted onnx model on CPU
    - There are two images in this repo, your file should run onnx model on these images and verify if the model outputs correct class id and class name
        - n01440764_tench belongs to class id 0
        - n01632777_axolotl belongs to class id 26
    - Test should report failure in case the outputs are not correct and pipeline should report the failed test cases    
- model.py with follwoing classess/functionalities, make them separate classes:
    - Onnx Model loading and prediction call
    - Pre-processing of the Image
- Things needed to deploy the code to banana dev
- test_server.py | codebase to make a call to the model deployed on banana dev
    - This should accept the path of image and return/print the name of class the image belongs to
    - And also accept a flag to run preset custom tests, where you make calls to the banana deployed models using the two images and verify results like you are expected to do in test_onnx.py
    - also report time one banana dev call takes
- Github Actions pipeline that can do following:
    - Pre-Deployment Pipeline:
        - Build the docker image
        - Run Onnx converter to see if it works
        - Run tests cases on converted ONNNX model
    - Post-Deployment Pipeline:
        - cron job scheduled using Github action
        - basically for health check, calls the test_server.py to run automated tests
        - run the tests every 24 hours, and report the time taken by the banana dev call
- Readme File | which has steps to run/use all the deliverables with proper details, such that a person who has no prior information of this repo can understand and run this easily with no blockers.
- Hint: You need to use Deploy From Github option, and make a fork of their public template.
    - Follow steps mentioned in the banana dev documentation


 ### Evaluation Criteria
 - *Python* best practices
 - Completeness: Did you include all features?
 - Correctness: Does the solution (all deliverables) work in sensible, thought-out ways?
 - Maintainability: Is the code written in a clean, maintainable way?
 - Testing: Is the solution adequately tested?
 - Documentation: Is the codebase well-documented and has proper steps to run any of the deliverables?

 
**Things which are very important and will be considered during evaluation**
- Your test_server.py should be properly implemented, we will use that to test your final deployment.
- Don't deploy PyTorch Model, you need to convert the PyTorch Model to ONNX first and use that in the deployment.
- Code Formating and Documentation.
- Proper use of Git and Github Actions.
- Meaningful and good commits, we will monitor commit history.
- Don't delete the other branch you make all the commits too.
    - Your flow can be you make changes in a new branch made from main.
    - And then in the end when you are done with changes you merge the new branch using proper protocols in main for banana dev deployment

**Extra Points:**
- Make pre-processing steps part of Onnx File [<name_of_model>.onnx file], needs to be done during onnx conversion, instead of implementing them in the code inside app.py.
- Create Github Actions pipeline such that if Docker Image successfully builds and test cases pass it automatically merges the PR into main.

**Note:**
- You would need to enter the debit/credit card details on banana.dev platform but you will not be charged.
- They would charge 1$ for testing of card, which gets refunded automatically by banana dev.
- You get 1 hour of Free GPU and unlimited CPU calls.
- Ideally, you should be charged zero USD for this assignment unless their is some bug on banana platform or the codebase you develop.
- Incase, some mishap occurs because of banana dev or your mistake and you are charged extra amount MTailor is not accountable for that.