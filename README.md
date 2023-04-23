
# TABLE IMAGE OCR Tool

This TABLE IMAGE OCR Tool is designed to read images and extract text using both Azure and AWS OCR clients. It processes the image, stores the detected text and their bounding boxes, and finally saves the extracted data in an Excel file. This tool also handles overlapping text areas and separates text if necessary.

## Setup
Setup Python3.x.

```Python
$ pip install opencv-python pandas numpy \
azure-cognitiveservices-vision-computervision \
boto3
```


## Usage/Examples
1. Replace the placeholders in "ImageToExcel" for Azure and AWS credentials with your own.py:
```Python
azure_ocr_client = AzureOCR(
    "YOUR_SUBSCRIPTION_KEY",
    "YOUR_ENDPOINT",
)

aws_ocr_client = AwsOCR(
    "YOUR_AWS_ACCESS_KEY_ID", 
    "YOUR_WS_SECRET_ACCESS_KEY", 
    "YOUR_REGION"
)
```

2. Run the script with the required command line arguments.

```Python
python table_image_ocr.py <number_of_tables> <path_to_image>
```
For example:
```Python
python table_image_ocr.py 2 example_image.jpg
```


<number_of_tables> indicates the number of tables arranged vertically. For example, if there are two tables arranged vertically in one image, the value will be 2. The number of tables arranged horizontally is not counted. For example, if there are two tables arranged in the left and right, the value will be 1.

3. The script will process the image and save the extracted data in an Excel file, as well as generate additional files containing the Azure and AWS OCR results.


## Output

The script will generate the following output files:

- *_withBoundingBox.jpg: Image with bounding boxes drawn for Azure (in red) and AWS (in green) OCR results.
- *.xlsx: Excel file with the extracted data.
- *_azure.txt: Text file containing the Azure OCR results (bounding boxes and text).
- *_azure.json: JSON file containing the Azure OCR results.
- *_aws.txt: Text file containing the AWS OCR results (bounding boxes and text).
- *_aws.json: JSON file containing the AWS OCR results.
## Limitation
- The script assumes a certain structure of tables and text in the image. If your input image has a different structure, the script may require modifications to work correctly.

## Correct Tilt

If you want to correct the tilt of an image before OCR,

```Python
python CorrectImageTilt.py <path_to_image_origin> <path_to_image_save> 
```

For example:
```Python
python CorrectImageTilt.py image_path_before_correct jpg save_path_after_correct
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

