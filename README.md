# 0. summary
https://jaeyoungstudio.notion.site/585eaa15093144028dacd69758348434?pvs=4

# 1. yolov8 smoking object detection training
- model: yolov8 model
- enviornment: google colab, Jetson Nano
- used object class: 'Cigarette', 'Person', 'Smoke', 'Vape', 'smoking'
- optimization: tensorRT(quantization) VS onnx(CPU optimization)
- result
[train result]  
![image](https://github.com/IntelliHuman/time-specific-yolov8-object-detection/assets/61938029/652d1039-5879-43be-b159-1dc88094df7e)
[validation result]  
<img width="1037" alt="image" src="https://github.com/IntelliHuman/time-specific-yolov8-object-detection/assets/61938029/c2c5a381-13cc-40bf-922f-94746ca7b878">  




# yolov8_used_csv
yolov8_used_csv
- used trained smoking detection model weights
- yolo predict model="C:\Users\rlati\Downloads\validation_results\content\runs\detect\yolov8n_custom\weights\best.pt" source=0 save=True
