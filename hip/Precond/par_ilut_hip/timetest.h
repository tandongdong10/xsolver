#ifndef timetest_h
#define timetest_h
#include<sys/time.h>
#include<stdio.h>
struct timeval time1;
struct timeval time2;
int timerecordlength=0;
const int Maxlength=300;
long long int timerecordarray[Maxlength];
void record_time(char*mark=NULL)
{
/*if(mark!=NULL)
{
printf("%s\n",mark);
}

if(timerecordlength==0)
{gettimeofday(&time1,NULL);
timerecordlength++;
}
else if(timerecordlength<Maxlength)
{
	gettimeofday(&time2,NULL);
    timerecordarray[timerecordlength]=(time2.tv_sec - time1.tv_sec)*1000000+ (time2.tv_usec - time1.tv_usec);
    printf("%lld \n",timerecordarray[timerecordlength]);
    timerecordlength++;
    
    gettimeofday(&time1,NULL);
    
}
else{
printf("record too many times!!!\n");
}*/
}
void printrecordtime()
{	
	printf("printrecordtime!\n");
	for(int i=0;i<timerecordlength;i++)
    {
    printf("%d %lld\n",i, timerecordarray[i]);
    }

}


#endif
