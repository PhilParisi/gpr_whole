#ifndef CUBLASHANDLER_H
#define CUBLASHANDLER_H


class cublasHandler : public cudaHandler
{
public:

private:
  /*!
   * \brief creaton of an instance of cublasHandler is not allowed.
   * it is intended to be used only with static members.
   */
  cublasHandler();

  static cublasHandle_t _handle; ///<  a p
};

#endif // CUBLASHANDLER_H
