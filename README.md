* Install the required libraries by using the below command
```console
pip install -r requirements.txt
```

* Connect to the redis server by using the below command *(we are assuming that you already have redis)*
```console
redis-server conf/redis/redis.conf
```

* Create and activate conda environment by using the below commands
```console
conda create --name SIC python=3.9
source activate SIC
```

* Set **IS_EMPATHY** to <mark>False</mark> for the empathetic robot or <mark>True</mark> for the non-empathetic robot.
* Run script to start the experiment.
