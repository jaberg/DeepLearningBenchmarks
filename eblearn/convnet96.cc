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

int main(int argc, char **argv) { // regular main without gui
  init_drand(92394); // initialize random seed

  intg n_examples = 100; // maximum training set size: 60000
  idxdim dims(1,96,96); // get order and dimensions of sample
  
  //! create 1-of-n targets with target 1.0 for shown class, -1.0 for the rest
  idx<t_net> targets = create_target_matrix(10, 1.0);
  idx<t_net> inputs(n_examples, 96, 96);

  parameter<t_net> theparam(6000); // create trainable parameter
  lenet5<t_net> l5(theparam, 96, 96, 7, 7, 3, 3, 7, 7, 3, 3, 120, 10);
    // TODO: use an all-to-all connection table in second layer convolution
    // Because that's what the other packages implement.
  supervised_euclidean_machine<t_net, ubyte> thenet(
          (module_1_1<t_net>&)l5,
          targets,
          dims);
  supervised_trainer<t_net, ubyte,ubyte> thetrainer(thenet, theparam);
  classifier_meter trainmeter, testmeter;
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

  state_idx<t_net> dummy_input(1, 96, 96); 
  double t = time_time();
  for (intg j = 0; j < n_examples; ++j)
  {
	thetrainer.learn_sample(dummy_input, j%10, gdp);
        // TODO: iterate over mock dataset to simulate more realistic
        // memaccess pattern
    }
#ifdef USED_IPP
  cout << "ConvMed\teblearn{ipp}\t" << n_examples / (time_time() - t) << endl;
#else
  cout << "ConvMed\teblearn\t" << n_examples / (time_time() - t) << endl;
#endif
  return 0;
}
