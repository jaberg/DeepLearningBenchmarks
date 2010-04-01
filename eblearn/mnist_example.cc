#include "libeblearn.h"
#include <time.h>
#include <sys/time.h>

using namespace std;
using namespace ebl; // all eblearn objects are under the ebl namespace

static double time_time() // a time function like time.time()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000.0;
}

typedef double t_net;

// argv[1] is expected to contain the directory of the mnist dataset
#ifdef __GUI__
MAIN_QTHREAD() { // this is the macro replacing main to enable multithreaded gui
#else
int main(int argc, char **argv) { // regular main without gui
#endif
  cerr << "* MNIST demo: learning handwritten digits using the eblearn";
  cerr << " C++ library *" << endl;
  if (argc != 2) {
    cerr << "Usage: ./mnist <my mnist directory>" << endl;
    eblerror("MNIST path not specified");
  }
  init_drand(time(NULL)); // initialize random seed

  intg trsize = 10000; // maximum training set size: 60000
  intg tesize = 10000; // maximum testing set size:  10000

  //! load MNIST datasets: trize for training set and tesize for testing set
  mnist_datasource<t_net, ubyte,ubyte> 
      train_ds(argv[1], "train", trsize),
      test_ds(argv[1], "t10k", tesize);

  //! create 1-of-n targets with target 1.0 for shown class, -1.0 for the rest
  idx<t_net> targets = create_target_matrix(1+idx_max(train_ds.labels), 1.0);

  //! create the network weights, network and trainer
  cerr << "creating idxdim: " << endl;
  idxdim dims(train_ds.sample_dims()); // get order and dimensions of sample
  cerr << "creating theparam: " << endl;
  parameter<t_net> theparam(60000); // create trainable parameter
  cerr << "creating l5: " << endl;
  lenet5<t_net> l5(theparam, 32, 32, 5, 5, 2, 2, 5, 5, 2, 2, 120, targets.dim(0));
  //TODO: Consider using net_nn_cscsc directly rather than lenet5

  cerr << "creating thenet: " << endl;
  supervised_euclidean_machine<t_net, ubyte> thenet((module_1_1<t_net>&)l5, targets, dims);
  cerr << "creating thetrainer: " << endl;
  supervised_trainer<t_net, ubyte,ubyte> thetrainer(thenet, theparam);
  //supervised_trainer_gui<t_net> stgui; // the gui to display supervised_trainer

  //! a classifier-meter measures classification errors
  classifier_meter trainmeter, testmeter;

  //! initialize the network weights
  forget_param_linear fgp(1, 0.5);
  thenet.forget(fgp);

  // learning parameters
  gd_param gdp(/* double leta*/ 0.0001,
         /* double ln */  0.0,
         /* double l1 */  0.0,
         /* double l2 */  0.0,
         /* int dtime */  0,
         /* double iner */0.0,
         /* double a_v */ 0.0,
         /* double a_t */ 0.0,
         /* double g_t*/  0.0);
  infer_param infp;

  int use_hessian = 0;
  // estimate second derivative on 100 iterations, using mu=0.02
  if (use_hessian)
  {
      cerr << "Computing second derivatives on MNIST dataset: " << endl;
      thetrainer.compute_diaghessian(train_ds, 100, 0.02);
  }

  //code borrowd from libeblearn/include/ebl_trainer.hpp
  ubyte lab;
  thetrainer.init(train_ds, &trainmeter);
  // training on lowest size common to all classes (times # classes)
  // now do training iterations
  //cerr << "... Training network from " << train_ds.get_lowest_common_size() << endl;
  double t = time_time();
  train_ds.fprop(*thetrainer.input, thetrainer.label);
  lab = thetrainer.label.get();
  //int J = train_ds.get_lowest_common_size();
  int J = 2000;
  for (intg j = 0; j < J; ++j) {
	//train_ds.fprop(*thetrainer.input, thetrainer.label);
	//lab = thetrainer.label.get();
	thetrainer.learn_sample(*thetrainer.input, lab, gdp);
	// use energy as distance for samples probabilities to be used
	///// train_ds.set_answer_distance(energy.x.get());
	//      log.update(age, output, label.get(), energy);
	//train_ds.next_train();
    }
#ifdef __IPP__
    cout << "lenet5\teblearn{ipp}\t" << J / (time_time() - t) << endl;
#else
    cout << "lenet5\teblearn\t" << J / (time_time() - t) << endl;
#endif
  return 0;
}


#if 0
  for (int i = 0; i < 100; ++i) {
    double t = time_time();
    cerr << "Training... " << endl;
    thetrainer.train(train_ds, trainmeter, gdp, 1);                // train
    cerr << "Training took" << t - time_time() << "seconds" << endl;
    cerr << "Testing on train... " << endl;
    thetrainer.test(train_ds, trainmeter, infp);           // test
    cerr << "Testing on test... " << endl;
    thetrainer.test(test_ds, testmeter, infp);                     // test
    //stgui.display_datasource(thetrainer, test_ds, infp, 10, 10); // display
    //stgui.display_internals(thetrainer, test_ds, infp, 2);       // display
    if (use_hessian)
        thetrainer.compute_diaghessian(train_ds, 100, 0.02); // recompute 2nd der
    cerr << "Iteration took" << t - time_time() << "seconds" << endl;
  }
#endif
