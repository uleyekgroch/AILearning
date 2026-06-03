# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/mayAi/AILearning_v0527/ai-learning-cpp/build-server/_deps/crow-src")
  file(MAKE_DIRECTORY "D:/mayAi/AILearning_v0527/ai-learning-cpp/build-server/_deps/crow-src")
endif()
file(MAKE_DIRECTORY
  "D:/mayAi/AILearning_v0527/ai-learning-cpp/build-server/_deps/crow-build"
  "D:/mayAi/AILearning_v0527/ai-learning-cpp/build-server/_deps/crow-subbuild/crow-populate-prefix"
  "D:/mayAi/AILearning_v0527/ai-learning-cpp/build-server/_deps/crow-subbuild/crow-populate-prefix/tmp"
  "D:/mayAi/AILearning_v0527/ai-learning-cpp/build-server/_deps/crow-subbuild/crow-populate-prefix/src/crow-populate-stamp"
  "D:/mayAi/AILearning_v0527/ai-learning-cpp/build-server/_deps/crow-subbuild/crow-populate-prefix/src"
  "D:/mayAi/AILearning_v0527/ai-learning-cpp/build-server/_deps/crow-subbuild/crow-populate-prefix/src/crow-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/mayAi/AILearning_v0527/ai-learning-cpp/build-server/_deps/crow-subbuild/crow-populate-prefix/src/crow-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/mayAi/AILearning_v0527/ai-learning-cpp/build-server/_deps/crow-subbuild/crow-populate-prefix/src/crow-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
