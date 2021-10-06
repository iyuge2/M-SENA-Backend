conda activate sena
export FLASK_APP=app.py
export FLASK_ENV=production
# export FLASK_ENV=development
# export APP_SETTINGS='config.ProductionConfig'
# export APP_SETTINGS='config.DevelopmentConfig'
# export DATABASE_URL='mysql://sena:wasdwasd@localhost/sena'
python app.py