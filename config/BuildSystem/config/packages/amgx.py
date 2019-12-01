import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = ''
    self.versionname      = ''
    self.gitcommit         = 'master'
    self.download         = ['git://https://github.com/petsc/AMGX']
    self.functions        = []
    self.includes         = ['amgx_c.h']
    self.liblist          = [['libamgx.a']]
    self.precisions       = ['double']
    self.cxx              = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.cuda           = framework.require('config.packages.cuda',self)
    self.deps           = [self.mpi,self.cuda]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

#    for place,item in enumerate(args):
#      if item.find('CMAKE_C_COMPILER') >= 0:
#        # findCUDA sets ccbin to the C compiler, force the C++ compiler instead
#        # this prevents the error
#        # base/src/amg_signal.cu(221): error: union "sigaction::<unnamed>" has no member "__sigaction_handler"
#        self.framework.pushLanguage('C++')
#        args[place]='-DCMAKE_C_COMPILER="'+self.framework.getCompiler()+'"'
#        self.framework.popLanguage()
    self.framework.pushLanguage('C++')
    args.append('-DCUDA_NVCC_FLAGS="-ccbin '+self.framework.getCompiler()+'"')
    self.framework.popLanguage()
    return args
