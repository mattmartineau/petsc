import config.compile.processor

class Preprocessor(config.compile.processor.Processor):
  '''The C++ preprocessor'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CXXCPP', 'CPPFLAGS', '.cpp', '.cc')
    return

class Compiler(config.compile.processor.Processor):
  '''The C++ compiler'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CXX', ['CXXFLAGS', 'CXX_CXXFLAGS'], '.cc', '.o')
    self.requiredFlags[-1] = '-c'
    self.outputFlag        = '-o'
    self.flagsName.extend(Preprocessor(argDB).flagsName)
    return

  def getTarget(self, source):
    '''Return None for header files'''
    import os

    base, ext = os.path.splitext(source)
    if ext in ['.h', '.hh']:
      return None
    return base+'.o'

  def getCommand(self, sourceFiles, outputFile = None):
    '''If no outputFile is given, do not execute anything'''
    if outputFile is None:
      return 'true'
    return config.compile.processor.Processor.getCommand(self, sourceFiles, outputFile)

class Linker(config.compile.processor.Processor):
  '''The C++ linker'''
  def __init__(self, argDB):
    compiler        = Compiler(argDB)
    config.compile.processor.Processor.__init__(self, argDB, ['CXX_LD', 'LD', compiler.name], 'LDFLAGS', '.o', '.a')
    self.outputFlag = '-o'
    if self.name == compiler.name:
      self.flagsName.extend(compiler.flagsName[:-1])
    return

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      return self.argDB['LIBS']
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')

  def getTarget(self, source, shared):
    import os
    import sys

    base, ext = os.path.splitext(source)
    if shared:
      return base+'.so'
    if sys.platform[:3] == 'win' or sys.platform == 'cygwin':
      return base+'.exe'
    return base
