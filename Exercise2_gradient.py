import numpy as np
import math as mt

'''
author : Danie Basulto
mail : basultobd@gmail.com

for more information about the Gradient descent 
see https://en.wikipedia.org/wiki/Gradient_descent#Computational_examples

'''

#gradient of the function
def evaluate_f_gradient(x_value, y_value):
    return np.array([ x_value*( 400*mt.pow( x_value,2 ) - 400*y_value +2 ) - 2 , 200*( y_value - mt.pow( x_value,2 ) ) ])

#original function
def eval_f(x_value, y_value):
    return 100*mt.pow( y_value - mt.pow( x_value, 2), 2 ) + mt.pow( 1 - x_value , 2)

def backtracking( alpha , rho, c, x_old, p_k_direction_vector ):

    '''

    This code use the the Armijo condition
    see wolfe rules for more info 
    https://en.wikipedia.org/wiki/Wolfe_conditions

    '''

    #calculates the new point x_k+1
    direction = alpha*p_k_direction_vector
    x_new = x_old + direction
    x_new_value = x_new[0]
    y_new_value = x_new[1]
    left_inequality_side = eval_f( x_new_value, y_new_value )

    #calculates tangent line at x_k in the direction of p
    x_old_value = x_old[0]
    y_old_value = x_old[1]
    gradient = evaluate_f_gradient(x_old_value, y_old_value) 
    right_inequality_side= eval_f( x_old_value, y_old_value ) + c*alpha*np.inner(gradient, p_k_direction_vector) 

    #if the nrew point is under the reduced "tangent line" at x_k+1 in the direction of p_k
    while left_inequality_side >= right_inequality_side:

        #calculates the new point x_k+1
        direction = alpha*p_k_direction_vector
        x_new = x_old + direction
        x_new_value = x_new[0]
        y_new_value = x_new[1]
        left_inequality_side = eval_f( x_new_value, y_new_value )

        #calculates tangent line at x_k in the direction of p_k
        x_old_value = x_old[0]
        y_old_value = x_old[1]
        gradient = evaluate_f_gradient(x_old_value, y_old_value) 
        right_inequality_side= eval_f( x_old_value, y_old_value ) + c*alpha*np.inner(gradient, p_k_direction_vector)

        #Reduce the alpha value
        alpha = alpha * rho

    return alpha


def main():


    old_x = np.array([10,10])

    #initial point
    new_x = np.array([2,3])

    alpha0 = 20
    c = 0.001
    rho = 0.5

    #max difference between x_k+1 and x_k
    precision = 0.00001 

    #sucetion of x_k points
    calculated_points_list = []

    #sucetion of f(x_k) values
    points_evaluated_list = []

    x_initial_value = new_x[0]
    y_initial_value = new_x[1]

    #evaluate the new point in the f(x,y) function
    f_evaluated = eval_f(x_initial_value, y_initial_value)

    #number of x_k's calculated
    number_iterations = 0

    calculated_points_list.append(("iteration: " + str(number_iterations),new_x))
    points_evaluated_list.append(("iteration: " + str(number_iterations),f_evaluated))

    #save the difference between vectors
    difference_vector = abs(new_x - old_x)

    #if all the values in the difference vector are greater than precision
    isValiddifference = (difference_vector >= precision ).any()


    while ( isValiddifference ):

        old_x = new_x
        x_old_value = old_x[0]
        y_old_value = old_x[1]

        #evaluate the x_k point in the gradient
        gradient = evaluate_f_gradient( x_old_value, y_old_value ) 

        #calculates the new alpha
        alpha = backtracking( alpha0, rho, c, old_x, -p_direction )
        alpha_mul_gradient = alpha*gradient

        #solve the Ax = b system. A = jacobian, x = sk_vector, b = eval_h
        new_x = old_x - alpha_mul_gradient

        x_new_value = new_x[0]
        y_new_value = new_x[1]

        f_evaluated = eval_f(x_new_value, y_new_value)

        number_iterations += 1

        calculated_points_list.append(("iteration: " + str(number_iterations),new_x))
        points_evaluated_list.append(("iteration: " + str(number_iterations),f_evaluated))

        difference_vector = abs(new_x - old_x)

        isValiddifference = (difference_vector >= precision ).any()

    #print the first and final point, and the value of f(x) in that points
    for i,(point,point_evaluated) in enumerate(zip(calculated_points_list,points_evaluated_list)):
        isFirstValue = ( i == 0 )
        isLastValue = ( i == len(calculated_points_list)-1 )
        if( isFirstValue or isLastValue ):
            print( "\n" )
            print("Point: " + str(point) )
            print("Point evaluated: " + str(point_evaluated) )


if __name__ == '__main__':
    main()