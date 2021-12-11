#!/bin/sh
cd src/tasks/task-1-eda/anpr_wpodnet_paddleocr &&
export FLASK_APP=app &&
export FLASK_ENV=development &&
flask run