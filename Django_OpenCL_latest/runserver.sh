cd ~/.virtualenvs/Django_VirtualEnv
. bin/activate
cd /home/nvidia/Downloads/Aviage_Xavier_11_October-20181011T152048Z-001/Aviage_Xavier_11_October/Aviage_Fuel_Problem/Teuvonet_Jetson_Site
PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}} python manage.py runserver &
sleep 6
chromium-browser 127.0.0.1:8000 --no-sandbox
while true; do 
PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}} python manage.py runserver
sudo fuser -k 8000/tcp
sleep 2
done
