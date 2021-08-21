#!/usr/bin/perl

use strict;
use warnings;
use File::Path 'rmtree';

my @projects = @ARGV;

foreach my $this_project (@projects) {
       if(-d $this_project) {
           my @files = <$this_project/*>;
           foreach my $file (@files) {
               if($file !~ m#/(?:(?:config\.ini)|(?:program)|(?:backup))$#) {
                   rmtree($file);
                   print $file."\n";
                }
           }
       } else {
           warn "$this_project is not a folder!";
       }
}
