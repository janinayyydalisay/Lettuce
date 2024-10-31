from .lettuce_predictor import predict_lettuce1
import io
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.http import JsonResponse
import base64


def login_page(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        print(user,"sample")
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            return redirect('login')
    return render(request,'login.html')

def logout_view(request):
    logout(request)
    return redirect('login')  # Redirect to a desired page after logout


def signup_page(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        User.objects.create_user(username=username, password=password)
        return redirect('/login')
    return render(request,'signup.html')

def home(request):
    return render(request,'home.html')

# Example usage in Django view
def predict(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        lettuce = predict_lettuce1(io.BytesIO(image_file.read()))
        return render(request, 'getImage.html', {'lettuce': lettuce})
    return render(request, 'getImage.html')

from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('lettuce.h5')

# Define the class names associated with the animals
class_names = ['Bacterial', 'Downymildew', 'Healthy', 'Powdery', 'Septoria', 'Sheperd', 'Viral', 'Wilt']


# Function to predict animal in an image
def predict_lettuce(frame):
    try:
        # Resize and preprocess the image
        frame = cv2.resize(frame, (224, 224))  # Resize to the model's input size
        img_array = image.img_to_array(frame)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image to [0,1]

        # Make the prediction
        prediction = model.predict(img_array)
        output_index = np.argmax(prediction)
        predicted_class = class_names[output_index]  # Get the predicted class name
        predicted_prob = float(np.max(prediction))   # Get the probability of the predicted class

        print(f"Predicted Class: {predicted_class}, Probability: {predicted_prob}")
        return predicted_class, predicted_prob

    except Exception as e:
        print("Error predicting lettuce:", str(e))
        return None, None


def process_video_frame(request):
    if request.method == 'POST':
        # Process the video frame data here
        # Extract the frame data from the POST request
        frame_data = request.POST.get('frame_data')

        # Decode the base64-encoded frame data
        _, encoded_data = frame_data.split(',')
        decoded_data = base64.b64decode(encoded_data)

        # Convert the decoded data to a numpy array
        nparr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Predict animal
        predicted_lettuce, predicted_prob = predict_lettuce(frame)

        # Draw a rectangle around the predicted animal on the canvas
        if predicted_lettuce is not None:
            height, width, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)

        # Return the response
        response_data = {
            'lettuce': predicted_lettuce,
            'probability': predicted_prob
        }
        return JsonResponse(response_data)

    return render(request, 'video.html')
