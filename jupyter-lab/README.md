jupyter-lab --generate-config
jupyter server password
vi ~/.jupyter/jupyter_lab_config.py
    c.LabServerApp.open_browser = False
    c.ServerApp.allow_root = True
    c.ServerApp.ip = '0.0.0.0'
    c.ServerApp.port = 8888