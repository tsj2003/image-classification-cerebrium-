**Deploy Classification Neural Network on Serverless GPU platform of Cerebrium**


You have to deploy a Machine Learning Model on Cerebrium [https://www.cerebrium.ai/].
You have to use GitHub for the codebase.

**Assignment Protocols**
- We expect it to take ~4 hours, with an extra 15 min for clear loom explanation(s)
    - The assessment is timeboxed at 5 hours total in a single block. So please plan accordingly
- You can only use Python as a programming Language
- You cannot take help from any other person
    - But you can use google to search for references
- Record a 5-10 mins of code walkthrough of the work you have done. You can use Loom Platform (https://www.loom.com) to record the video
    - A live demo of each of the features mentioned below:
        - Cerebrium platform deployment page
        - Other scripts as required in "Deliverable" section below
        - Show all steps of Cerebrium deployment running successfully, and what you think as a pre-requisite to trigger deployment
    - Code overview of each of those features:
        - Why did you implement it that way?
        - Is there any way you would improve it?
    - Explain what tests you have developed and why
    - Explain what parts of the assessment are completed and what is missing?
    - Make sure to submit the screen recording link in the submission after you are done recording
    - Please note that the free plan on Loom only allows for videos up to 5 minutes in length. As such, you may need to record two separate 5-minute videos

**Cerebrium Details:**
- You need to use custom Docker Image based deployment. Any submission which is not based on Dockerfile will be rejected.
- This example from Cerebrium explains how to use Docker Image [https://github.com/CerebriumAI/examples/tree/master/2-advanced-concepts/5-dockerfile]


**Model Details:**
- Model is designed to perform classification on an input Image
- Model will be used in production where one would expect answers within 2-3 seconds
- PyTorch Implementation of model is present in pytorch_model.py, and weights can be downloaded from this link: https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0
- The model is trained on ImageNet Dataset [https://www.image-net.org]
- The input to the model is an image of size 224x224, and the output is the array with probabilities for each class.
- The length of the output array is equal to the number of classes [1000] in the ImageNet dataset.
- There are two images in this repo:
    - n01440764_tench belongs to class id 0
    - n01667114_mud_turtle belongs to class id 35

Model is trained on images with specific pre-processing steps, e.g. you need to do the following pre-processings on the image before passing it to the model. A function (preprocess_numpy) is implemented in the model class which performs the necessary pre-processing on the image, and at the end of pytorch_model.py you can see how to use the model on an image file to get inference.
- Convert to RGB format if needed. The model accepts the image in RGB format, not in BGR. Code will never throw errors for this so keep an eye on the library you use to load image.
- Resize to 224x224 (use bilinear interpolation)
- Divide by 255
- Normalize using mean values for each channel [RGB][0.485, 0.456, 0.406] and standard deviation values for each channel [RGB] [0.229, 0.224, 0.225]
    - subtract mean and divide standard deviation per channel

**Deliverable**
- convert_to_onnx.py | codebase to convert the PyTorch Model to the ONNX model
- model.py with the following classes/functionalities, make their separate classes:
    - Onnx Model loading and prediction call
    - Pre-processing of the Image [Sample code provided in pytorch_model.py]
- test.py | codebase to test the code/model written. This should test everything one would expect for ML Model deployment.
- Things needed to deploy the code to the Cerebrium
- test_server.py | codebase to make a call to the model deployed on the Cerebrium (Note: This should test deployment not something on your local machine)
    - This should accept the path of the image and return/print the id of the class the image belongs to
    - And also accept a flag to run preset custom tests, something like test.py but uses deployed model.
    - Add more tests to test the Cerebrium as a platform. Anything to monitor the deployed model.
- Readme File | which has steps to run/use all the deliverables with proper details, such that a person who has no prior information about this repo can understand and run this easily with no blockers.

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
- Proper use of Git.
- Meaningful and good commits, we will monitor commit history.

**Extra Points:**
- CI pipeline to test Docker Image builds succesfully everytime we push a new commit to repo.
- Make pre-processing steps part of Onnx File [<name_of_model>.onnx file], which needs to be done during onnx conversion, instead of implementing them in the code inside app.py.

**Note:**
- You get 30 USD of free credits on Cerebrium on new signup.
    - This is more than enough for this task. You would at max spend 2-3 USD from free credits.
- In case, you add your credit/debit card on the platform (which is not needed) and some mishap occurs, and you are charged an extra amount MTailor is not accountable for that.
