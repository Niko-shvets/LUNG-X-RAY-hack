# app.py

from flask import Flask, request, render_template, send_from_directory, Response,jsonify
import json
import cv2
#car model classification
from classifier import ensemble
from PIL import Image
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

__author__ = 'vlasov'

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("service.html")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/upload", methods=['GET', 'POST'])
def upload():
	target = os.path.join(APP_ROOT, 'images/')
	print(target)
	if not os.path.isdir(target):
			os.mkdir(target)
	else:
		print("Couldn't create upload directory: {}".format(target))
	print(request.files.getlist("file"))
	labels = []
	names = []
	for upload in request.files.getlist("file"):
		# print(upload)
		print("{} is the file name".format(upload.filename))
		filename = upload.filename
		destination = "".join([target, filename])# "/"
		# print ("Accept incoming file:", filename)
		# print ("Save it to:", destination)
		upload.save(destination)
		label = ensemble(destination)

		labels.append(label)
		names.append(filename)
	
	response = {"Labels":labels,"id":names}
	response = json.dumps(response, ensure_ascii=False)
	print('response|',response)
	
	# return send_from_directory("images", filename, as_attachment=True)
	return Response(response=response, status=200, mimetype="application/json")	
	# return render_template("complete.html", image_name=filename)



# @app.route("/upload", methods=["POST"])
# def upload():
# 	target = os.path.join(APP_ROOT, 'images/')
# 	print(target)
# 	if not os.path.isdir(target):
# 			os.mkdir(target)
# 	else:
# 		print("Couldn't create upload directory: {}".format(target))
# 	print(request.files.getlist("file"))
# 	labels = []
# 	names = []
# 	for upload in request.files.getlist("file"):
# 		print(upload)
# 		print("{} is the file name".format(upload.filename))
# 		filename = upload.filename
# 		destination = "".join([target, filename])# "/"
# 		print ("Accept incoming file:", filename)
# 		print ("Save it to:", destination)
# 		upload.save(destination)
# 		label = ensemble(destination)

# 		labels.append(label)
# 		names.append(filename)
	
# 	response = {"Labels":labels,"id":names}
# 	response = json.dumps(response, ensure_ascii=False)
# 	print('response|',response)
	
# 	# return send_from_directory("images", filename, as_attachment=True)
# 	return render_template("complete.html", image_name=filename)

# @app.route('/upload', methods=['GET', 'POST'])
# def service():
# 	if request.method == 'POST':
# 		filenames = request.files.getlist("file")

# 		print(request.files.get('file', ''))
# 		print(np.fromstring(request.data, np.uint8))
# 		labels = []
# 		names = []
# 		for filename in filenames:
# 			print(filename.filename)
# 			label = ensemble(filename.filename)
# 			labels.append(label)
# 			names.append(filename.filename)

			
# 		response = {"Labels":labels,"id":names}
# 		response = json.dumps(response, ensure_ascii=False)


# 		return Response(response=response, status=200, mimetype="application/json")	
# 	return render_template("service.html")


# We only need this for local development.
if __name__ == '__main__':
	app.run(port=4555, debug=True)
