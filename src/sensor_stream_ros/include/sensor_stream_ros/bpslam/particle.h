#ifndef PARTICLE_H
#define PARTICLE_H

//stl
#include <deque>
#include <unordered_set>
#include <atomic>
//ros
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
//local
#include <sensor_stream/blockgpr.h>
#include <sensor_stream_ros/bpslam/tf_chain_buffer.h>


namespace ss{ namespace bpslam {

class ParticleData{
public:
  std::string type;
  uint16_t version;
};

typedef std::shared_ptr<ParticleData> ParticleDataPtr;

template <typename T>
class Particle: public std::enable_shared_from_this<Particle<T>>
{
public:
  // typedefs
  typedef std::shared_ptr<Particle<T>> Ptr;
  typedef std::shared_ptr<const Particle<T>> ConstPtr;

  Particle();
  /*!
   * \brief adds a child to the particle
   * \param child a ParticlePtr to the child particle you want to add
   */
  void addChild(std::shared_ptr<Particle> child);
  /*!
   * \brief removes the particle from the tree, deletes itself from it's parents children,
   * and calls removeFromTree on all of its children.
   */
  void removeFromTree();

  /*!
   * \brief getChildren
   * \return a List of the children of the particle
   */
  const std::unordered_set<Particle::Ptr>getChildren(){return children_;}
  /*!
   * \brief getParent
   * \return a ParticlPtr to the parent of the particle
   */
  Particle::Ptr getParent(){return parent_;}
  /*!
   * \brief getParent
   * \return a ParticlPtr to this o
   */
  Particle::Ptr getThis(){return std::enable_shared_from_this<Particle<T>>::shared_from_this();}
  /*!
   * \brief getId
   * \return the unique id number of the particle
   */
  size_t getId(){return id_;}

  /*!
   * \brief casts the datapointer as a T type and returns it
   * \return
   */
  std::shared_ptr<T> getData();

  /*!
   * \brief is this the root particle of the tree
   * \return
   */
  bool isRoot(){return parent_==nullptr;}

  /*!
   * \brief is this particle a leaf (no children) particle
   * \return
   */
  bool isLeaf(){return children_.size()==0;}
  // operators

  bool operator==(Particle<T> & other);


  TfChainBuffer::Ptr tf_buffer;
protected:

  std::shared_ptr<T> particle_data_;
  std::shared_ptr<Particle> parent_;
  std::unordered_set<Particle::Ptr> children_;
  size_t id_;
  static std::atomic_ulong particles_created_;
};

}}

#endif // PARTICLE_H
