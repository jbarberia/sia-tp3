.PHONY: test clean

test:
	@echo "Running tests..."
	@python3 -m pytest test
	
clean:
	@echo "Cleaning up..."	
	@rm -rf */__pycache__/
	@rm -rf */*.pyc
	@rm -f *.png
	@rm -f *.jpg

reset_out:
	@rm -rf out/*

ejercicio_3_digitos:
	python main.py './config/ejercicio_3_digitos.toml'

ejercicio_4:
	python3 main.py 'config/ejercicio_4_config00_sgd.toml'
	python3 main.py 'config/ejercicio_4_config01_momentum.toml'
	python3 main.py 'config/ejercicio_4_config02_rmsprop.toml' 
	python3 main.py 'config/ejercicio_4_config03_adagrad.toml'
	python3 main.py 'config/ejercicio_4_config04_adam.toml'   

ejercicio_4_lr:
	python main.py 'config/ejercicio_4_config05_lr_bs.toml'
	python main.py 'config/ejercicio_4_config06_lr_bs.toml'
	python main.py 'config/ejercicio_4_config07_lr_bs.toml'
	python main.py 'config/ejercicio_4_config08_lr_bs.toml'

