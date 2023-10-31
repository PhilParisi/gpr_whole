#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "particle.h"

namespace ss{ namespace bpslam {

template <typename T>
std::atomic_ulong Particle<T>::particles_created_(0);  // initialaze static member var

template <typename T>
Particle<T>::Particle(){
  id_=particles_created_;
  particles_created_++;
  particle_data_.reset(new T());
  tf_buffer.reset(new TfChainBuffer);
}

template <typename T>
void Particle<T>::addChild(std::shared_ptr<Particle> child){
  assert(child!=nullptr);
  child->parent_=getThis();
  child->tf_buffer->setParent(tf_buffer); // set the child's tf buffer's parent to this's buffer
  children_.insert(child);
}

template <typename T>
void Particle<T>::removeFromTree(){
  if(parent_!=nullptr){
    parent_->children_.erase(getThis());
    parent_.reset();
  }
  for (auto child : children_) {
    child->removeFromTree();
  }
}

template <typename T>
bool Particle<T>::operator==(Particle<T> & other){
  return id_==other.id_;
}

template <typename T>
std::shared_ptr<T> Particle<T>::getData(){
  return particle_data_;
}

}}

#endif // PARTICLE_H
