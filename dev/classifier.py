import tensorflow as tf
import time, shutil, os

import cag_input_data_mm as cdata
import inspect_classes as inclass

# Architecture
n_hidden_1 = 50
n_hidden_2 = 50
output_dim = 2 # 2 classes for now: Caucasian & AA
input_dim = 2 # using 2-d ae encodings as input for now

# Parameters
learning_rate = 0.01
training_epochs = 301
batch_size = 1000
display_step = 25

def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    #return tf.nn.relu(tf.matmul(input, W) + b)
    return tf.nn.sigmoid(tf.matmul(input, W) + b)

def inference(x):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, [input_dim, n_hidden_1], [n_hidden_1])
     
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [n_hidden_1, n_hidden_2], [n_hidden_2])
     
    with tf.variable_scope("output"):
        output = layer(hidden_2, [n_hidden_2, output_dim], [output_dim])

    return output

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)    
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation", accuracy)
    return accuracy

if __name__ == '__main__':
    
    val_size = 450
    test_size = 1000
    input_type = 'ae'
    cag = inclass.read_data_classifier(input_type=input_type,
                                       validation_size=val_size,
                                       test_size=test_size)

    if os.path.exists("mlp_logs/"):
        shutil.rmtree("mlp_logs/")

    with tf.Graph().as_default():

        with tf.variable_scope("mlp_model"):

            x = tf.placeholder("float", [None, input_dim])
            y = tf.placeholder("float", [None, output_dim])


            output = inference(x)

            cost = loss(output, y)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = training(cost, global_step)

            eval_op = evaluate(output, y)

            summary_op = tf.summary.merge_all()

            saver = tf.train.Saver()

            sess = tf.Session()

            summary_writer = tf.summary.FileWriter("mlp_logs/",
                                                   graph_def=sess.graph_def)

            
            init_op = tf.global_variables_initializer()

            sess.run(init_op)

            # saver.restore(sess, "mlp_logs/model-checkpoint-66000")


            # Training cycle
            for epoch in range(training_epochs):

                avg_cost = 0.
                total_batch = int(cag.train.num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    minibatch_x, minibatch_y = cag.train.next_batch(batch_size)
                    # Fit training using batch data
                    sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                    # Compute average loss
                    avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y})/total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

                    accuracy = sess.run(eval_op, feed_dict={x: cag.validation.images, y: cag.validation.labels})

                    print("Validation Error:", (1 - accuracy))

                    summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                    summary_writer.add_summary(summary_str, sess.run(global_step))

                    saver.save(sess, "mlp_logs/model-checkpoint", global_step=global_step)


            print("Optimization Finished!")


            accuracy = sess.run(eval_op, feed_dict={x: cag.test.images, y: cag.test.labels})

            print("Test Accuracy:", accuracy)
