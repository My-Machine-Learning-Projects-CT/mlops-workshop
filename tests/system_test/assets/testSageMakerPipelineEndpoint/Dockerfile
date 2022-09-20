# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.8

# Copy the earlier created requirements.txt file to the container
COPY requirements.txt ./

# Install the python requirements from requirements.txt
RUN python3.8 -m pip install --no-cache-dir -r requirements.txt

# Copy the earlier created lambda.py file to the container
COPY lambda.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["lambda.handler"]