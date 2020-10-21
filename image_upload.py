from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64
from model import demo_modi

app = Flask(__name__)

# upload HTML rendering


@app.route('/upload')
def render_file():
    return render_template('upload.html')

# process file upoading


@app.route('/fileUpload', methods='POST')
def upload_file():
    # saved the image
    f = request.files['image']
    upload_path = './uploads/' + secure_filename(f.filename)
    f.save(upload_path)

    # get the 3d_render_image
    masked_img = get_mask3d_render_image(upload_path)
    img_string = base64.b64encode(masked_img)
    return jsonify(data=img_string)


def get_mask3d_render_image(img_path):
    mask3d_img = demo_modi.mask3d_render(img_path)
    return mask3d_img


def encoded_image_to_base64(img_path):
    image = open(img_path, 'rb')
    img_read = image.read()
    image_64_encode = base64.b64encode(img_read)
    return image_64_encode


if __name__ == "__main__":
    # 서버 실행
    app.run(debug=True)
    # encoded_image = encoded_image_to_base64('./uploads/go2.jpg')
    # print(encoded_image)
