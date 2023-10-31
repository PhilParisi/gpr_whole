#include "abstract_block.h"
namespace ss {



AbstractBlock::AbstractBlock(){

}


BlockParams::Ptr AbstractBlock::getParams(){
    if(params_==nullptr){
        throw std::out_of_range("No BlockParams has been set!  Call setBlockParam first or use the Block(std::shared_ptr<BlockParams> params) constructor");
    }
    return params_;
}

}
