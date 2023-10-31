#ifndef DEVHOSTDATA_H
#define DEVHOSTDATA_H

template <class T>
class DevHostData
{
public:
  DevHostData();
  ~DevHostData();
  void initHost(size_t n = 1);
  void initDev(size_t n = 1);
  void dev2host();
  void host2dev();

private:
  T * host_data;
  T * dev_data;
};

#endif // DEVHOSTDATA_H
