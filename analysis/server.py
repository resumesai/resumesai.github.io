import time  # Importing the time library to check the time of code execution
import sys  # Importing the System Library
import os
import socket

from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
from flask_cors import CORS, cross_origin
import traceback
from assoc import get_rater_rules
from dtree import make_tree

server = Flask(__name__)
CORS(server)


@server.route("/")
def index():
    return render_template("index.html")  # The template html


@server.route("/assoc/<rater>")
def get(rater):
    try:
        print("rater = ", rater)
        ridx = int(rater)
        res = get_rater_rules(ridx)
        print(res)
        return res
    except Exception as ex:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(ex)})


@server.route("/dtree/<rater>")
def tree(rater):
    try:
        print("rater = ", rater)
        ridx = int(rater)
        res = make_tree(ridx)
        print(res)
        return res
    except Exception as ex:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(ex)})


if __name__ == "__main__":
    server.run()
