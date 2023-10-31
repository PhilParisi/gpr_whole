#ifndef PF_BASE_H
#define PF_BASE_H
//stl
#include <deque>
#include <mutex>
//local
#include "particle.hpp"

namespace ss{ namespace bpslam {

template <typename T>
class PFBase
{
public:
  PFBase();  ///< \brief default constructor,  creates it's own root particle
  PFBase(typename Particle<T>::Ptr root); ///< \brief  create a new PFBase with by setting your own root particle
  /*!
   * \brief addLeaf add a leavf to a given paerent particle
   * \param leaf the node you want to add
   * \param parent the parent you want to add it to
   */
  void addLeaf(typename Particle<T>::Ptr leaf,typename Particle<T>::Ptr parent);
  /*!
   * \brief creates a new "fresh" leaf and adds it the the parent
   * \param parent the particle you want to add the leaf to
   */
  void addNewLeaf(typename Particle<T>::Ptr parent);
  /*!
   * \brief pops the element at the front of the leaf queue
   * \return the element that was removed from the front of the queue
   */
  void removeLeaf(typename Particle<T>::Ptr particle_ptr);
  /*!
   * \brief removes all ancestor particles that have no remaining children.
   */
  void removeAncestors(typename Particle<T>::Ptr particle_ptr, bool remove_children = true);
  /*!
   * \brief returns the element at the front of the leafQueue
   * \return
   */
  typename Particle<T>::Ptr leafQuequeFront();

  // getters/setters ~~~~~~~~~~~~~~~~~~~~~

  //!
  //! \brief Gets the particle at the root of the particle tree
  //! \return the root particle
  //!
  typename Particle<T>::Ptr rootParticlePtr(){return root_particle_;}
  /*!
   * \brief getLeafQueue
   * \return the current queue of leaf particles
   * \todo figure out why everything breaks when I switch this function to return by reference
   */
  const std::unordered_set<typename Particle<T>::Ptr> getLeafQueue(){ std::lock_guard<std::mutex> lk(leaf_mutex_); return leaf_queue_;}
protected:
  std::mutex leaf_mutex_;
  typename Particle<T>::Ptr root_particle_;
  std::unordered_set<typename Particle<T>::Ptr> leaf_queue_;
};



}}

#endif // PF_BASE_H
