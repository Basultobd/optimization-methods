import numpy as np
import math as mt

'''
author : Danie Basulto
mail : basultobd@gmail.com

for more information about the Newton methd for optimization
see https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization

'''
#hessian of the function
def eval_f_hessian( x_value, y_value ):
	return np.array([[-400*y_value+1200*mt.pow(x_value,2)+2,-400*x_value],[-400*x_value,200]])

#gradient of the function
def eval_f_gradient(x_value, y_value):
    return np.array([ x_value*( 400*mt.pow( x_value,2 ) - 400*y_value +2 ) - 2 , 200*( y_value - mt.pow( x_value,2 ) ) ])

#original function
def eval_f_function(x_value , y_value):
	return 100*mt.pow( ( y_value + mt.pow( x_value,2 ) ) , 2) + mt.pow(1-x_value, 2)

#vector used for avoid calculate the inverse of the jacobian
def calculate_sk_vector(hessian,f_gradient_evaluated):
	s_k = np.linalg.solve(hessian, f_gradient_evaluated)
	return s_k


def main():

	old_f = np.array([10,10])

	#initial point
	new_f = np.array([2,3])

	#max difference between x_k+1 and x_k
	precision = 0.00001 

	#sucetion of x_k points
	calculated_points_list = []

	#sucetion of f(x_k) values
	points_evaluated_list = []

	x_initial_value = new_f[0]
	y_initial_value = new_f[1]

	#evaluate the new point in the original function
	f_original_eval = eval_f_function(x_initial_value, y_initial_value)

	#number of x_k's calculated
	number_iterations = 0

	calculated_points_list.append(("iteration: " + str(number_iterations),new_f))
	points_evaluated_list.append(("iteration: " + str(number_iterations),f_original_eval))

	#save the difference between vectors
	difference_vector = abs(new_f - old_f)

	#if all the values in the difference vector are greater than precision
	isValiddifference = ( difference_vector >= precision ).any()

	while( isValiddifference ):

		old_f = new_f

		x_old_value = old_f[0]
		y_old_value = old_f[1]

		#evaluate in the hessian
		hessian = eval_f_hessian(x_old_value, y_old_value)

		#evaluate in the gradient
		f_gradient_evaluated = eval_f_gradient(x_old_value, y_old_value)

		#calculate the s_k vector
		sk_vector = calculate_sk_vector(hessian, -f_gradient_evaluated)

		#calculates the new x_k+1 point
		new_f = old_f + sk_vector

		x_new_value = new_f[0]
		y_new_value = new_f[1]

		f_original_eval = eval_f_function(x_new_value, y_new_value)

		number_iterations += 1

		calculated_points_list.append(("iteration: " + str(number_iterations),new_f))
		points_evaluated_list.append(("iteration: " + str(number_iterations),f_original_eval))

		difference_vector = abs(new_f - old_f)

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

