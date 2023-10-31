#ifndef PF_BASE_HPP
#define PF_BASE_HPP
#include "pf_base.h"

namespace ss{ namespace bpslam {

template <typename T>
PFBase<T>::PFBase()
{
  std::lock_guard<std::mutex> lk(leaf_mutex_);
  root_particle_.reset(new Particle<T>);
  leaf_queue_.insert(root_particle_);
}

template <typename T>
PFBase<T>::PFBase(typename Particle<T>::Ptr root){
  std::lock_guard<std::mutex> lk(leaf_mutex_);
  root_particle_ = root;
  leaf_queue_.insert(root_particle_);
}

template <typename T>
void PFBase<T>::addLeaf(typename Particle<T>::Ptr leaf, typename Particle<T>::Ptr parent){
  assert(leaf!=nullptr);
  assert(parent!=nullptr);
  std::lock_guard<std::mutex> lk(leaf_mutex_);
  parent->addChild(leaf);
  leaf_queue_.insert(leaf);
}

template <typename T>
void PFBase<T>::addNewLeaf(typename Particle<T>::Ptr parent){
  std::lock_guard<std::mutex> lk(leaf_mutex_);
  typename Particle<T>::Ptr leaf;
  leaf.reset(new ss::bpslam::Particle<T>);
  addLeaf(leaf,parent);
}

template <typename T>
void PFBase<T>::removeLeaf(typename Particle<T>::Ptr particle_ptr){
  std::lock_guard<std::mutex> lk(leaf_mutex_);
  //particle_ptr->removeFromTree();
  leaf_queue_.erase(particle_ptr);
}

template <typename T>
void PFBase<T>::removeAncestors(typename Particle<T>::Ptr particle_ptr, bool remove_children){
  typename Particle<T>::Ptr parent = particle_ptr->getParent();
  if(remove_children  || particle_ptr->isLeaf() ){
    particle_ptr->removeFromTree();
    removeAncestors(parent,false);
  }
}

template <typename T>
typename Particle<T>::Ptr PFBase<T>::leafQuequeFront(){
  return *leaf_queue_.begin();
}

}}
#endif // PF_BASE_H
