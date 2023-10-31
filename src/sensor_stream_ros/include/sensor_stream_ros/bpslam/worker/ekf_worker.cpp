#include "ekf_worker.h"


namespace ss{ namespace bpslam {

EKFWorker::EKFWorker(){

}

EKFWorker::EKFWorker(ParticlePtr_t particle, EKFParams::ConstPtr params){
  input.particle=particle;
  input.params=params;

}

void EKFWorker::run(){
  if(input.particle->getParent()==nullptr){
    return;  // base case
  }
  for(size_t i=0; i<6 ;i++)
    output.velocity_error[i]=0;
  sample();
  integrate();
  return;
}

void EKFWorker::sample(){
  auto inital_odom = input.particle->getParent()->getData()->getFinalHypothesis();

  unsigned seed = input.particle->getId();//std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  for(size_t i = 0 ; i<6; i++){
    double stdev = std::sqrt(inital_odom->twist.covariance[idx::rowMajor(i,i,6)]);
    std::normal_distribution<double> distribution(0.0,stdev* input.params->uncertainty_multiplier);
    output.velocity_error[i] = distribution(generator);
    }
  return;
//  Eigen::VectorXd mean(6);
//  Eigen::MatrixXd covar(6,6);
//  for(size_t i = 0 ; i<6; i++){
//    for(size_t j = 0 ; j<6; j++){
//      covar(i,j) = inital_odom->twist.covariance[idx::rowMajor(i,j,6)];
//    }
//  }

//  Eigen::EigenMultivariateNormal<double,6> normX_solver(mean,covar);
//  Eigen::Matrix<double,6,-1> samples = normX_solver.samples(1);
//  for(size_t i = 0 ; i<6; i++){
//    output.velocity_error[i] = samples(i,0);
//    if(abs(output.velocity_error[i])>2){
//      output.velocity_error[i]=abs(samples(i,0))/samples(i,0)*2;
//      std::cerr << "very large displacement detected" << std::endl;
//    }
//  }
}

void EKFWorker::integrate(){
  auto first = input.particle->getData()->nav.odom_front;
  auto last = input.particle->getData()->nav.odom_back;
  auto data_ptr = input.particle->getData();
  data_ptr->nav.hypothesis.clear();
  data_ptr->nav.hypothesis.push_back(integrate(input.particle->getParent()->getData()->getFinalHypothesis(),*first));
  for(auto it = first; it != last ; it++){
    nav_msgs::Odometry::Ptr odom = integrate(data_ptr->nav.hypothesis.back(),*it);
    data_ptr->nav.hypothesis.push_back(odom);
    geometry_msgs::TransformStamped transform;
    transform.header =                   odom->header;
    transform.child_frame_id =           odom->child_frame_id;
    transform.transform.rotation =       odom->pose.pose.orientation;
    transform.transform.translation.x =  odom->pose.pose.position.x;
    transform.transform.translation.y =  odom->pose.pose.position.y;
    transform.transform.translation.z =  odom->pose.pose.position.z;
    input.particle->tf_buffer->setTransform(transform,"EKFWorker");
  }
}

void calcDisplacement(double & out_v, double & displacement, float  v_error, double  dt){
  out_v += v_error;
  displacement = out_v * dt;
}

nav_msgs::Odometry::Ptr EKFWorker::integrate(nav_msgs::Odometry::ConstPtr last, nav_msgs::Odometry::ConstPtr reference){
  ros::Duration dt = reference->header.stamp - last->header.stamp;
  nav_msgs::Odometry::Ptr out(new nav_msgs::Odometry);
  *out=*reference;
  tf2::Quaternion rotation;
  tf2::fromMsg(reference->pose.pose.orientation,rotation);
  tf2::Transform transform(rotation);
  tf2::Vector3 vechicle_displacement_vect(0,0,0);
  for(auto i : input.params->random_vars){
    switch (i) {
      case idx::x_linear:
      calcDisplacement(out->twist.twist.linear.x,
                       vechicle_displacement_vect[i],
                       output.velocity_error[i],
                       dt.toSec());
      break;
      case idx::y_linear:
      calcDisplacement(out->twist.twist.linear.y,
                       vechicle_displacement_vect[i],
                       output.velocity_error[i],
                       dt.toSec());
      break;
      case idx::z_linear:
      calcDisplacement(out->twist.twist.linear.z,
                       vechicle_displacement_vect[i],
                       output.velocity_error[i],
                       dt.toSec());
      break;
      default:
        std::cerr << "you have entered an usupported random var parameter" << std::endl;
    }
  }
  tf2::Vector3 odom_displacement_vect = transform(vechicle_displacement_vect);  // transform our output to the odom frame
  for(auto i : input.params->random_vars){
    switch (i) {
    case idx::x_linear:
      out->pose.pose.position.x = last->pose.pose.position.x + odom_displacement_vect.x();
    break;
    case idx::y_linear:
      out->pose.pose.position.y = last->pose.pose.position.y + odom_displacement_vect.y();
    break;
    case idx::z_linear:
      out->pose.pose.position.z = last->pose.pose.position.z + odom_displacement_vect.z();
    break;
    default:
      std::cerr << "you have entered an usupported random var parameter" << std::endl;
    }
  }

  return out;
}

}}
