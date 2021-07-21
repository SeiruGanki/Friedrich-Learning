import numpy

# generate uniform distributed points in a domain [a1,b1]X[a2,b2]X...X[ad,bd]
# domain_intervals = [[a1,b1],[a2,b2],...,[ad,bd]]
def generate_uniform_points_in_cube(domain_intervals,N):
    d = domain_intervals.shape[0]
    points = numpy.zeros((N,d))
    for i in range(d):
        points[:,i] = numpy.random.uniform(domain_intervals[i,0],domain_intervals[i,1],(N,))
    return points

def generate_uniform_points_in_cylinder(domain_intervals,N):
    d = domain_intervals.shape[0]
    r = domain_intervals.shape[1]
    points = numpy.zeros((N,d))
    polar_axis = numpy.zeros((N,d))
    polar_axis[:,0] = numpy.random.uniform(domain_intervals[0,0],domain_intervals[0,1],(N,))
    polar_axis[:,1] = numpy.random.uniform(0,r**2,(N,))
    polar_axis[:,2] = numpy.random.uniform(0,2*numpy.pi,(N,))
    points[:,0] = polar_axis[:,0]
    points[:,1] = numpy.sqrt(polar_axis[:,1]) * numpy.cos(polar_axis[:,2])
    points[:,2] = numpy.sqrt(polar_axis[:,1]) * numpy.sin(polar_axis[:,2])
    return points

def generate_uniform_points_in_circle(domain_intervals,N): # generate points at T=0 in a circle
    d = domain_intervals.shape[0]
    r = domain_intervals.shape[1]
    points = numpy.zeros((N,d))
    polar_axis = numpy.zeros((N,d))
    polar_axis[:,1] = numpy.random.uniform(0,r**2,(N,))
    polar_axis[:,2] = numpy.random.uniform(0,2*numpy.pi,(N,))
    points[:,1] = numpy.sqrt(polar_axis[:,1]) * numpy.cos(polar_axis[:,2])
    points[:,2] = numpy.sqrt(polar_axis[:,1]) * numpy.sin(polar_axis[:,2])
    return points

# generate uniform distributed points in a domain [T0,T1]X[a1,b1]X[a2,b2]X...X[ad,bd]
# domain_intervals = [[a1,b1],[a2,b2],...,[ad,bd]]
def generate_uniform_points_in_cube_time_dependent(domain_intervals,time_interval,N):
    d = domain_intervals.shape[0]
    points = numpy.zeros((N,d+1))
    points[:,0] = numpy.random.uniform(time_interval[0],time_interval[1],(N,))
    for i in range(0,d):
        points[:,i+1] = numpy.random.uniform(domain_intervals[i,0],domain_intervals[i,1],(N,))
    return points

# generate uniform distributed points in the sphere {x:|x|<R}
# input d is the dimension
def generate_uniform_points_in_sphere(d,R,N):
    points = numpy.random.normal(size=(N,d))
    scales = (numpy.random.uniform(0,R,(N,)))**(1/d)
    for i in range(N):
        points[i,:] = points[i,:]/numpy.sqrt(numpy.sum(points[i,:]**2))*scales[i]
    return points

# generate uniform distributed points in the sphere [T0,T1]X{x:|x|<R}
# input d is the dimension
def generate_uniform_points_in_sphere_time_dependent(d,R,time_interval,N):
    points = numpy.zeros((N,d+1))
    points[:,0] = numpy.random.uniform(time_interval[0],time_interval[1],(N,))
    points[:,1:] = numpy.random.normal(size=(N,d))
    scales = (numpy.random.uniform(0,R,(N,)))**(1/d)
    for i in range(N):
        points[i,1:] = points[i,1:]/numpy.sqrt(numpy.sum(points[i,1:]**2))*scales[i]
    return points


# generate uniform distributed points on the boundary of domain [a1,b1]X[a2,b2]X...X[ad,bd]
# domain_intervals = [[a1,b1],[a2,b2],...,[ad,bd]]
def generate_uniform_points_on_cube(domain_intervals,N_each_face):
    d = domain_intervals.shape[0]
    if d == 1:
        return numpy.array([[domain_intervals[0,0]],[domain_intervals[0,1]]])
    else:
        points = numpy.zeros((2*d*N_each_face,d))
        for i in range(d):
            points[2*i*N_each_face:(2*i+1)*N_each_face,:] = numpy.insert(generate_uniform_points_in_cube(numpy.delete(domain_intervals,i,axis=0),N_each_face), i, values=domain_intervals[i,0]*numpy.ones((1,N_each_face)), axis = 1)
            points[(2*i+1)*N_each_face:(2*i+2)*N_each_face,:] = numpy.insert(generate_uniform_points_in_cube(numpy.delete(domain_intervals,i,axis=0),N_each_face), i, values=domain_intervals[i,1]*numpy.ones((1,N_each_face)), axis = 1)
        return points

# generate uniform distributed points on the boundary of time-dependent domain [T0,T1]X[a1,b1]X[a2,b2]X...X[ad,bd]
# and at the initial slice slice {t=T0}X[a1,b1]X[a2,b2]X...X[ad,bd]
# whole_intervals = [[T0,T1]X[a1,b1],[a2,b2],...,[ad,bd]]
def generate_uniform_points_on_cube_time_dependent(domain_intervals,time_interval,N_each_face,N_initial_time_slice, time_condition_type = 'initial'):
    d = domain_intervals.shape[0]
    whole_intervals = numpy.zeros((d+1,2))
    whole_intervals[0,:] = time_interval
    whole_intervals[1:,:] = domain_intervals
    points_bd = numpy.zeros((2*d*N_each_face,1+d))
    points_int = numpy.zeros((N_initial_time_slice,1+d))
    for i in range(1,1+d):
        points_bd[2*(i-1)*N_each_face:(2*i-1)*N_each_face,:] = numpy.insert(generate_uniform_points_in_cube(numpy.delete(whole_intervals,i,axis=0),N_each_face), i, values=whole_intervals[i,0]*numpy.ones((1,N_each_face)), axis = 1)
        points_bd[(2*i-1)*N_each_face:2*i*N_each_face,:] = numpy.insert(generate_uniform_points_in_cube(numpy.delete(whole_intervals,i,axis=0),N_each_face), i, values=whole_intervals[i,1]*numpy.ones((1,N_each_face)), axis = 1)
    if time_condition_type == 'initial':
        points_int = numpy.insert(generate_uniform_points_in_cube(numpy.delete(whole_intervals,0,axis=0),N_initial_time_slice), 0, values=whole_intervals[0,0]*numpy.ones((1,N_initial_time_slice)), axis = 1)
    elif time_condition_type == 'terminal':
        points_int = numpy.insert(generate_uniform_points_in_cube(numpy.delete(whole_intervals,0,axis=0),N_initial_time_slice), 0, values=whole_intervals[0,1]*numpy.ones((1,N_initial_time_slice)), axis = 1)
    return points_bd, points_int

# generate uniform distributed points on the boundary of domain {|x|<R}
def generate_uniform_points_on_sphere(d,R,N_boundary):
    if d == 1:
        return numpy.array([[-R],[R]])
    else:
        points = numpy.zeros((N_boundary,d))
        for i in range(N_boundary):
            points[i,:] = numpy.random.normal(size=(1,d))
            points[i,:] = points[i,:]/numpy.sqrt(numpy.sum(points[i,:]**2))*R
        return points

# generate uniform distributed points on the boundary of time-dependent domain [T0,T1]X{|x|<R}
# except for the final slice {t=T1}X{|x|<R}
def generate_uniform_points_on_sphere_time_dependent(d,R,time_interval,N_boundary,N_initial_time_slice):
    if d == 1:
        return generate_uniform_points_on_cube_time_dependent(numpy.array([[-R,R]]),time_interval,int(round(N_boundary/2)), N_initial_time_slice, time_condition_type = 'initial')
    points_bd = numpy.zeros((N_boundary,d+1))
    points_int = numpy.zeros((N_initial_time_slice,d+1))
    points_int[:,0] = time_interval[0]*numpy.ones(N_initial_time_slice,)
    points_int[:,1:] = numpy.random.normal(size=(N_initial_time_slice,d))
    for i in range(N_boundary):
        points_bd[i,0] = numpy.random.uniform(time_interval[0],time_interval[1])
        points_bd[i,1:] = numpy.random.normal(size=(1,d))
        points_bd[i,1:] = points_bd[i,1:]/numpy.sqrt(numpy.sum(points_bd[i,1:]**2))*R
    points_int[:,0] = time_interval[0]*numpy.ones(N_initial_time_slice,)
    points_int[:,1:] = numpy.random.normal(size=(N_initial_time_slice,d))
    scales = (numpy.random.uniform(0,R,(N_initial_time_slice,)))**(1/d)
    for i in range(N_initial_time_slice):
        points_int[i,1:] = points_int[i,1:]/numpy.sqrt(numpy.sum(points_int[i,1:]**2))*scales[i]
    return points_bd, points_int

# generate a list of learning rates
def generate_learning_rates(highest_lr_pow,lowest_lr_pow,total_iterations,ratio_get_to_the_lowest,n_stage):
    lr = 10 ** lowest_lr_pow * numpy.ones((total_iterations,))
    lr_n = numpy.ceil(numpy.arange(0,n_stage+1)*(total_iterations*ratio_get_to_the_lowest/n_stage)).astype(numpy.int32)
    for i in range(n_stage):
        lr[lr_n[i]:lr_n[i+1]-1] = 10 ** (highest_lr_pow + (lowest_lr_pow-highest_lr_pow)/n_stage*i)
    return lr
