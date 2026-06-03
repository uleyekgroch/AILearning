
if(NOT "/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-subbuild/crow-populate-prefix/src/crow-populate-stamp/crow-populate-gitinfo.txt" IS_NEWER_THAN "/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-subbuild/crow-populate-prefix/src/crow-populate-stamp/crow-populate-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-subbuild/crow-populate-prefix/src/crow-populate-stamp/crow-populate-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout --config "advice.detachedHead=false" "https://github.com/CrowCpp/Crow.git" "crow-src"
    WORKING_DIRECTORY "/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/CrowCpp/Crow.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout v1.2.0 --
  WORKING_DIRECTORY "/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v1.2.0'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-src"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-subbuild/crow-populate-prefix/src/crow-populate-stamp/crow-populate-gitinfo.txt"
    "/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-subbuild/crow-populate-prefix/src/crow-populate-stamp/crow-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda/_deps/crow-subbuild/crow-populate-prefix/src/crow-populate-stamp/crow-populate-gitclone-lastrun.txt'")
endif()

