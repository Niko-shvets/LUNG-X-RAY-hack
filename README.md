дообучили 2 модели:
 - ResNet на tensorflow [ссылка на веса](https://drive.google.com/file/d/1VBXpbJk9izunRsaaWtIQUsFkgP7NCeSl/view?usp=sharing)
 - CheXNet (DenseNet обученная на датасете  chestX-ray-14) [ссылка на веса](https://drive.google.com/file/d/1GHEHli6vxIoHpqR8351sDrmXRtHtPwwe/view?usp=sharing)

В ноутбуке Network Ensemble.ipynb модели работают в ансамбле.
В файле Resnet_training.ipynb идет дообучение сети ResNet.

Api для обработки:
 - app.py
 - classifier.py для воспроизведения моделей
 - templates/ шаблоны html страниц 
