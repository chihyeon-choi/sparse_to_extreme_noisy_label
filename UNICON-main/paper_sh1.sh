python Train_cifar.py --dataset cifar10 --num_class 10 --noise_mode 'sym' --r 0.5
python Train_cifar.py --dataset cifar10 --num_class 10 --noise_mode 'sym' --r 0.9
python Train_cifar.py --dataset cifar10 --num_class 10 --noise_mode 'asym' --r 0.4

python Train_cifar_wo_cont.py --dataset cifar10 --num_class 10 --noise_mode 'sym' --r 0.5
python Train_cifar_wo_cont.py --dataset cifar10 --num_class 10 --noise_mode 'sym' --r 0.9
python Train_cifar_wo_cont.py --dataset cifar10 --num_class 10 --noise_mode 'asym' --r 0.4

python Train_cifar_wo_lu.py --dataset cifar10 --num_class 10 --noise_mode 'sym' --r 0.5
python Train_cifar_wo_lu.py --dataset cifar10 --num_class 10 --noise_mode 'sym' --r 0.9
python Train_cifar_wo_lu.py --dataset cifar10 --num_class 10 --noise_mode 'asym' --r 0.4

python Train_cifar_wo_lu_cont.py --dataset cifar10 --num_class 10 --noise_mode 'sym' --r 0.5
python Train_cifar_wo_lu_cont.py --dataset cifar10 --num_class 10 --noise_mode 'sym' --r 0.9
python Train_cifar_wo_lu_cont.py --dataset cifar10 --num_class 10 --noise_mode 'asym' --r 0.4