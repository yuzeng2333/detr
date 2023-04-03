#include <stdio.h>
#include <assert.h>

int mainQ(int k){
     assert (k>=0);
     assert (k<=30);
     
     int y = 0;
     int x = 0;
     int c = 0;

     // loop 1
     printf("k, c, x, y\n");     
     while(1){
	  //assert(4*x-(y*y*y*y)-2*(y*y*y)-(y*y) == 0);
      //assert(c <= k);
	  //%%%traces: int x, int y, int k

	  if (!(c < k)) break;
    
	  c = c +1 ;
	  y=y +1;
	  x=y*y*y+x;
    printf("%d, %d, %d, %d\n", k, c, x, y);    
     }
     return x;
}


int main(int argc, char **argv){
     mainQ(atoi(argv[1]));
     return 0;
}

