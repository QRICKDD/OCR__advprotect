import easyocr
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=r'C:\Users\zou_zheng\.EasyOCR\model')
result=reader.readtext()