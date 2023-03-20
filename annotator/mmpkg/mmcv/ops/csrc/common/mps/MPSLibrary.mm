#include "MPSLibrary.h"
#include "MPSDevice.h"

static std::unique_ptr<MPSLibraryManager> mps_library_manager=nullptr;

MPSLibraryManager* MPSLibraryManager::getInstance() {
  if(!mps_library_manager)
    mps_library_manager = std::unique_ptr<MPSLibraryManager>(new MPSLibraryManager());
  return mps_library_manager.get();
}

MPSLibraryManager::~MPSLibraryManager() {}

MPSLibraryManager::MPSLibraryManager() {}

bool MPSLibraryManager::hasLibrary(const std::string& name) {
  return _library_map.find(name) != _library_map.end();
}

MPSLibrary* MPSLibraryManager::getLibrary(const std::string& library_url) {
  if (_library_map.find(library_url) != _library_map.end()) {
    return _library_map[library_url].get();
  }
  _library_map.emplace(std::make_pair(
      library_url, std::unique_ptr<MPSLibrary>(MPSLibrary::createFromUrl(library_url))));
  return _library_map[library_url].get();
}

MPSLibrary* MPSLibraryManager::createLibraryFromSouce(const std::string& name,
                                                      const std::string& source) {
  NSString* ns_name = [NSString stringWithCString:name.c_str()];
  if (_library_map.find(name) != _library_map.end()) {
    NSLog(@"Library %@ already exist.", ns_name);
    return nullptr;
  }

  _library_map.emplace(
      std::make_pair(name, std::unique_ptr<MPSLibrary>(MPSLibrary::createFromSource(source))));
  return _library_map[name].get();
}

MPSLibrary* MPSLibrary::createFromUrl(const std::string& library_url) {
  MPSLibrary* library = new MPSLibrary();
  @autoreleasepool {
    NSError* error = nil;

    // load library and func
    NSString* utl_str = [NSString stringWithCString:library_url.c_str()];
    NSURL* metal_url = [NSURL fileURLWithPath:utl_str];
    library->_library = [at::mps::MPSDevice::getInstance()->device() newLibraryWithURL:metal_url
                                                                                 error:&error];
    if (library->_library == nil) {
      NSLog(@"Failed to find library, error %@.", error);
      exit(1);
    }
  }

  return library;
}

MPSLibrary* MPSLibrary::createFromSource(const std::string& sources) {
  MPSLibrary* library = new MPSLibrary();
  @autoreleasepool {
    NSError* error = nil;

    // load library and func
    NSString* code_str = [NSString stringWithCString:sources.c_str()];
    library->_library = [at::mps::MPSDevice::getInstance()->device() newLibraryWithSource:code_str
                                                                                  options:nil
                                                                                    error:&error];
    if (library->_library == nil) {
      NSLog(@"Failed to find library, error %@.", error);
      exit(1);
    }
  }

  return library;
}

MPSLibrary::~MPSLibrary() {
  [_library release];
  _library = nil;
}

MTLComputePipelineState_t MPSLibrary::getComputePipelineState(const std::string& function_name) {
  if (_pso_map.find(function_name) != _pso_map.end()) {
    return _pso_map[function_name];
  }

  MTLComputePipelineState_t pso;
  @autoreleasepool {
    NSError* error = nil;

    // create function
    NSString* function_name_str = [NSString stringWithCString:function_name.c_str()];
    id<MTLFunction> func = [_library newFunctionWithName:function_name_str];
    if (func == nil) {
      NSLog(@"Failed to created pipeline state object, error %@.", error);
      exit(1);
    }
    // create pipeline
    pso = [at::mps::MPSDevice::getInstance()->device() newComputePipelineStateWithFunction:func
                                                                                     error:&error];
    _pso_map.emplace(std::make_pair(function_name, pso));
  }
  return _pso_map[function_name];
}
