import numpy as np
import math as mt

'''
author : Danie Basulto
mail : basultobd@gmail.com

for more information about the Newton methd for optimization
see https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization

'''

#jacobian of the function
def eval_h_jacobian(x_value, y_value):
	return np.array([[2*mt.exp(2*x_value+y_value)-1,mt.exp(2*x_value+y_value)],[2*x_value,0]])

#evaluate in the h(x,y) function 
def eval_h_function(x_value,y_value):
	return np.array([mt.exp(2*x_value+y_value)-x_value, mt.pow(x_value,2)-1])

#original function. See https://en.wikipedia.org/wiki/Gradient_descent , solution of a non-linear system section
def eval_original_function(x_value, y_value):
    return 0.5*( mt.pow(mt.exp(2*x_value+y_value)-x_value,2) + mt.pow(mt.pow(x_value,2)-1,2))

#vector used for avoid calculate the inverse of the jacobian
def calculate_sk_vector(jacobian,h_evaluated):
	s_k = np.linalg.solve(jacobian, h_evaluated)
	return s_k


def main():

	old_h = np.array([10,10])

	#initial point
	new_h = np.array([1,1])

	#max difference between x_k+1 and x_k
	precision = 0.00001 

	#sucetion of x_k points
	calculated_points_list = []

	#sucetion of f(x_k) values
	points_evaluated_list = []

	x_initial_value = new_h[0]
	y_initial_value = new_h[1]

	#evaluate the new point in the original function
	f_original_eval = eval_original_function(x_initial_value, y_initial_value)

	#number of x_k's calculated
	number_iterations = 0

	calculated_points_list.append(("iteration: " + str(number_iterations),new_h))
	points_evaluated_list.append(("iteration: " + str(number_iterations),f_original_eval))

	#save the difference between vectors
	difference_vector = abs(new_h - old_h)

	#if all the values in the difference vector are greater than precision
	isValiddifference = ( difference_vector >= precision ).any()

	while( isValiddifference ):

		old_h = new_h

		x_old_value = old_h[0]
		y_old_value = old_h[1]

		#evaluate in the jacobian
		jacobian = eval_h_jacobian(x_old_value, y_old_value)

		#evaluate in the h(x,y) function
		eval_h = eval_h_function(x_old_value, y_old_value)

		#solve the Ax = b system. A = jacobian, x = sk_vector, b = eval_h
		sk_vector = calculate_sk_vector(jacobian, eval_h)

		#calculates the new x_k+1 point
		new_h = old_h - sk_vector

		x_new_value = new_h[0]
		y_new_value = new_h[1]

		f_original_eval = eval_original_function(x_new_value, y_new_value)

		number_iterations += 1

		calculated_points_list.append(("iteration: " + str(number_iterations),new_h))
		points_evaluated_list.append(("iteration: " + str(number_iterations),f_original_eval))

		difference_vector = abs(new_h - old_h)

		isValiddifference = (difference_vector >= precision ).any()


	#print the first and final points, also print the value of f(x) in that points
	for i,(point,point_evaluated) in enumerate(zip(calculated_points_list,points_evaluated_list)):
		isTheFirstValue = ( i == 0 )
		isTheLastValue = ( i == len(calculated_points_list) - 1 )
		if( isTheFirstValue or isTheLastValue ):
			print( "\n" )
			print("Point: " + str(point) )
			print("Point evaluated: " + str(point_evaluated) )

if __name__ == '__main__':
	main()

