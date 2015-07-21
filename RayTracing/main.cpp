#include <gl\glut.h>
#include <iostream>
#include <ctime>
#include "model.h"
#include "bvh.h"
#include "screen.h"
#include "traversal.h"
#include <thrust/device_ptr.h>
#include <cuda_runtime_api.h>

ScreenParams params;
bool* vhit;
float* vcolor;
void initFunc()															
{  
    glClearColor(AmbientRed, AmbientGrn, AmbientBlu, 0.0);							
    glShadeModel(GL_SMOOTH);											
    glClearDepth(1.0f);																			
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);													
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);				
}

void ReSizeGLScene(GLsizei width, GLsizei height)						
{
    if (height==0)																				
        height=1;									
    glViewport(0,0,width,height);											
    glMatrixMode(GL_PROJECTION);								
    glLoadIdentity();												
    gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,1000.0f);	
    glMatrixMode(GL_MODELVIEW);											
    glLoadIdentity();												
}

void keyboard(int key,int x,int y)
{
    switch(key)
    {
    case GLUT_KEY_UP:
        params.look_at.z+=0.3f;break;
    case GLUT_KEY_DOWN:
        params.look_at.z-=0.3f;break;
    case GLUT_KEY_LEFT:
        params.look_at.y-=0.3f;break;
    case GLUT_KEY_RIGHT:
        params.look_at.y+=0.3f;break;
    }
}

void myDisplay()														
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);					
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE);	

    glLoadIdentity();	
    gluLookAt( EYEX0, EYEY0, EYEZ0, params.look_at.x, params.look_at.y, params.look_at.z, 0, Upy, 0);
    glTranslatef(0.0f,0.0f,0.0f);
    long cnt=0;

    //	 ofstream myfile ("example1.txt");
    //	 if (!myfile.is_open())cout<<"the file is not open!"<<endl;	

    for (int i = 0; i < ScrHeight * ScrWidth; i++)
    {
        int x = i % ScrWidth;
        int y = i / ScrWidth;
        if (vhit[i])// && s==9 && t==16
        {
            cnt++;	
            //					myfile << vcolor[i][j]*1.1<<" "<<vcolor[i][j]*1.1<< " "<<vcolor[i][j]*1.1<<"\n";
            glColor3f(vcolor[i] * 1.1, vcolor[i] * 1.1,vcolor[i] * 1.1);
        }
        else
        {		
            //					myfile <<0<<" "<<0<< " "<<1<<"\n";
            glColor3f(0, 0, 1);
        }
        //float p0[3] = { pScrLeftDown[0]/2+j*mPixelRight[0]/2+i*mPixelUp[0]/2, pScrLeftDown[1]/2+j*mPixelRight[1]/2+i*mPixelUp[1]/2, pScrLeftDown[2]/2+j*mPixelRight[2]/2+i*mPixelUp[2]/2 };
        float3 p0 = params.screen_left_down + x * params.pixel_right + y * params.pixel_up;
        glBegin(GL_QUADS);
        glVertex3f(p0.x, p0.y, p0.z);
        glVertex3f(p0.x + params.pixel_right.x, p0.y + params.pixel_right.y, p0.z + params.pixel_right.z);
        glVertex3f(p0.x + params.pixel_right.x + params.pixel_up.x,
            p0.y + params.pixel_right.y + params.pixel_up.y,
            p0.z + params.pixel_right.z + params.pixel_up.z);
        glVertex3f(p0.x + params.pixel_up.x, p0.y + params.pixel_up.y, p0.z + params.pixel_up.z);
        glEnd();
    }
    //	 myfile.close();


    //myfile.open("example2.txt");
    // if (!myfile.is_open())cout<<"the file is not open!"<<endl;	
    // for (int i=0; i<ScrHeight; i++)
    //{
    //	for (int j=0; j<ScrWidth; j++)
    //	{
    //		int s=i/pixelsNums;
    //		int t=j/pixelsNums;
    //		if( s==9 && t==16){
    //			if(vhit[i][j] == true )
    //				myfile << vcolor[i][j]*1.1<<" "<<vcolor[i][j]*1.1<< " "<<vcolor[i][j]*1.1<<"\n";
    //			else
    //				myfile <<0<<" "<<0<< " "<<1<<"\n";
    //		}
    //	}
    // }
    // myfile.close();

    //cout<<"光线和场景相交的个数："<<cnt<<endl;
    glutSwapBuffers();
}

int main(int argc, char** argv)
{
    Model model = ReadModel("dragon2.ply");
    Model device_model = GetDeviceCopy(model);
    params = InitScreenParams();
    Bvh bvh = BuildDeviceBvh(device_model);
    //Bvh bvh = CreateBvh(model);
    vhit = new bool[ScrHeight * ScrWidth];
    vcolor = new float[ScrWidth * ScrHeight];
    for (int i = 0; i < ScrHeight * ScrWidth; ++i) 
    {
        vhit[i] = false;
        vcolor[i] = 0;
    }
    //CreateRays(params);
    GetColorAllPixels(params, device_model, bvh, vhit, vcolor);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB|GLUT_DEPTH );
    glutInitWindowSize(WinSizeW, WinSizeH);
    glutInitWindowPosition(0,0);
    glutCreateWindow("Galina Dragon");
    initFunc();
    glutDisplayFunc(myDisplay);

    // 	glutSpecialFunc(keyboard);
    // 	glutIdleFunc(myDisplay);

    glutReshapeFunc(ReSizeGLScene);
    glutMainLoop();  

    return 0;
}