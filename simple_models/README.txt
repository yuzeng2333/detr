I am going to try this approach:
1. create a series of simple models, each one does a simple task. With the outputs of all the simple models, it becomes much easier to find the equations
2. models
(1) predicts the largest degree for each variable. If successful, make a bigger model: predicts all the degress for each variable
(2) predicts the largest degree for all the equations, If successful: predicts all the degress in the equations

==================
transformer_512.py is a second version the transformer model I made. Its input is 512x6: a sequence of 512 elements. each element has a feature size of 6. This is because, the 512 elements (loop values) should be position-interchangable: arbitrarily change their positions should not change the outputs

