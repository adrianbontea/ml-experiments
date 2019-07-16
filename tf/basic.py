from base import ExperimentBase
import tensorflow as tf


class BasicTensorFlowExperiment(ExperimentBase):
    async def do_run_async(self):
        default_graph = tf.get_default_graph()
        # x and y variables added to the default graph
        x = tf.Variable(3, name="x")
        y = tf.Variable(5, name="y")

        print(f"x variable is part of the default graph: {x.graph == default_graph}")
        print(f"y variable is part of the default graph: {y.graph == default_graph}")

        # Try add f function and a new variable to a separate graph
        graph = tf.Graph()

        with graph.as_default():
            f = x * x * y + y + 2
            z = tf.Variable(3, name="z")

        print(f"f function is part of the default graph: {f.graph == default_graph}")
        print(f"f function is part of the new graph: {f.graph == graph}")

        print(f"z variable is part of the default graph: {z.graph == default_graph}")
        print(f"z variable is part of the new graph: {z.graph == graph}")

        # Only the new variable node, z will be added to the new graph since it's independent.
        # Even if defined in the scope of the new graph set as default,
        # the function will be added to the same graph as x and y variable nodes as it depends on them

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            init.run()
            result = f.eval()
            print(result)

        x = tf.placeholder(tf.int32)
        y = tf.Variable(7, name="y")
        f = x + y + 1

        with tf.Session() as sess:
            y.initializer.run()
            result = sess.run(f, feed_dict={x: [1, 2, 3]})
            print(result)
