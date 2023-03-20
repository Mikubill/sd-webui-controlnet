#ifndef _MPS_LIBRARY_H_
#define _MPS_LIBRARY_H_

#include <string>
#include <unordered_map>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
typedef id<MTLLibrary> MTLLibrary_t;
#else
typedef void* MTLComputePipelineState;
typedef void* MTLComputePipelineState_t;
typedef void* MTLLibrary;
typedef void* MTLLibrary_t;
#endif

class MPSLibrary {
 public:
  // disable constructor for singleton
  static MPSLibrary* createFromUrl(const std::string& library_url);
  static MPSLibrary* createFromSource(const std::string& source);
  ~MPSLibrary();

  MTLLibrary_t library() { return _library; }

  MTLComputePipelineState_t getComputePipelineState(
      const std::string& function_name);

 private:
  MTLLibrary_t _library;
  std::unordered_map<std::string, MTLComputePipelineState_t> _pso_map;
};

class MPSLibraryManager {
 public:
  // disable constructor for singleton
  MPSLibraryManager(const MPSLibraryManager&) = delete;
  MPSLibraryManager& operator=(const MPSLibraryManager&) = delete;
  MPSLibraryManager(MPSLibraryManager&&) = delete;
  MPSLibraryManager& operator=(MPSLibraryManager&&) = delete;

  static MPSLibraryManager* getInstance();

  bool hasLibrary(const std::string& name);

  MPSLibrary* getLibrary(const std::string& library_url);

  MPSLibrary* createLibraryFromSouce(const std::string& name,
                                     const std::string& sources);

  ~MPSLibraryManager();

 private:
  MPSLibraryManager();
  std::unordered_map<std::string, std::unique_ptr<MPSLibrary>> _library_map;
};
#endif
