import numpy as np
import math as mt

def eval_f_hessian( x_value, y_value ):
	return np.array([[-400*y_value+1200*mt.pow(x_value,2)+2,-400*x_value],[-400*x_value,200]])

def eval_f_gradient(x_value, y_value):
    return np.array([ x_value*( 400*mt.pow( x_value,2 ) - 400*y_value +2 ) - 2 , 200*( y_value - mt.pow( x_value,2 ) ) ])

def eval_f_function(x_value , y_value):
	return 100*mt.pow( ( y_value + mt.pow( x_value,2 ) ) , 2) + mt.pow(1-x_value, 2)

def calculate_sk_vector(hessian,f_gradient_evaluated):
	s_k = np.linalg.solve(hessian, f_gradient_evaluated)
	return s_k


def main():
	old_f = np.array([10,10])
	new_f = np.array([2,3])

	print("Funcion evaluada con el valor incial: " + str(eval_f_function(new_f[0],new_f[1])))

	iterations = []

	difference_vector = abs(new_f - old_f)
	precision = 0.00001

	j = 0;
	while((difference_vector >= precision).any()):
		old_f = new_f

		x_value = old_f[0]
		y_value = old_f[1]

		hessian = eval_f_hessian(x_value, y_value)
		f_gradient_evaluated = eval_f_gradient(x_value, y_value)
		f_gradient_evaluated = [-1*f_gradient_evaluated[0],-1*f_gradient_evaluated[1]]
		sk_vector = calculate_sk_vector(hessian, f_gradient_evaluated)
		new_f = old_f + sk_vector
		difference_vector = abs(new_f - old_f)
		iterations.append(new_f)

	for index in range(len(iterations)):
		print(iterations[index])

	print("\n Funcion evaluada con el valor final: " + str(eval_f_function(new_f[0],new_f[1])))

if __name__ == '__main__':
	main()

