#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <immintrin.h>
#include <sstream>
#include <thread>
#include <array>
#include <queue>

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include "imgui.h"
#include "imgui-SFML.h"

#include <glm/glm.hpp>
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/scalar_constants.hpp> // glm::pi
#include <glm/gtc/type_ptr.hpp>

namespace fs = std::experimental::filesystem;
namespace std {
    template <typename T>
    string to_string_with_precision(const T a_value, const int64_t n = 2)
    {
        ostringstream out;
        out.precision(n);
        out << fixed << a_value;
        return out.str();
    }

    string formatted_double(const double value, const int64_t n = 2)
    {
        string str_val = to_string_with_precision(value, n);
        int  outer = 0;
        bool decimal = false;
        for (auto iter = str_val.end(); iter > str_val.begin(); iter--)
        {
            if (decimal)
            {
                if (outer % 3 == 0 && outer != 0)
                    iter = str_val.insert(iter, ',');
            }
            else if (str_val[str_val.length() - outer - 1] == '.')
            {
                decimal = true;
                outer = -1;
            }
            outer++;
        }
        return str_val;
    }
}

int NUM_THREADS = std::thread::hardware_concurrency();
const int64_t VECWIDTH = 4;
const double pi = 2 * acos(0.0);
__m256i m_zeroi  = _mm256_set1_epi64x(0);
__m256i m_onesi  = _mm256_set1_epi64x(uint64_t(-1));

bool runSimulation = true;
bool drawPlane = true;
int global_tick = 0;

void drawSphere(double r, int lats, int longs) {
    int i, j = 0;
    for (i = 0; i <= lats; i++) {
        double lat0 = pi * (-0.5 + (double)(i - 1) / lats);
        double z0 = sin(lat0);
        double zr0 = cos(lat0);

        double lat1 = pi * (-0.5 + (double)i / lats);
        double z1 = sin(lat1);
        double zr1 = cos(lat1);

        glBegin(GL_QUAD_STRIP);
        for (j = 0; j <= longs; j++) {
            double lng = 2 * pi * (double)(j - 1) / longs;
            double x = cos(lng);
            double y = sin(lng);

            glNormal3f(x * zr0, y * zr0, z0);
            glVertex3f(r * x * zr0, r * y * zr0, r * z0);
            glNormal3f(x * zr1, y * zr1, z1);
            glVertex3f(r * x * zr1, r * y * zr1, r * z1);
        }
        glEnd();
    }
}

void glLoadMatrixf(glm::mat4 matrix)
{
    float* fM;
    fM = glm::value_ptr(matrix);
    glLoadMatrixf(fM);
}

int main()
{
    sf::Clock deltaClock;
    sf::Clock frameClock;

    sf::RenderWindow window(sf::VideoMode(1000, 1000), "ImGui + SFML = <3", sf::Style::Default, sf::ContextSettings(32));
    window.setFramerateLimit(60);
    ImGui::SFML::Init(window);

    sf::Vector2u windowSize = window.getSize();
    sf::Vector2i windowMiddle = sf::Vector2i(windowSize.x / 2, windowSize.y / 2);
    bool cursorGrabbed = true;
    sf::Vector2f mousePos = { 0.f,0.f };
    
    // Opengl stuff -------------------------------
    
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float cameraPitch = 0.f;
    float cameraYaw = 0.f;

    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraDirection = glm::normalize(cameraPos - cameraTarget);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDirection));
    cameraUp = glm::cross(cameraDirection, cameraRight);

    
    double frustRight = 1;
    double frustUp = frustRight * double(windowMiddle.y) / double(windowMiddle.x);
    double nearClip = 1.f;
    double fov = 70;
    double farClip = double(windowMiddle.x) / double(tan(fov * pi / 360));
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-frustRight, frustRight, -frustUp, frustUp, nearClip, farClip);
    glMatrixMode(GL_MODELVIEW);

    GLfloat black[] = { 0.0, 0.0, 0.0, 1.0 };
    GLfloat white[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat direction[] = { 0.0, 0.0, 1.0, 1.0 };

    glMaterialfv(GL_FRONT,  GL_AMBIENT_AND_DIFFUSE, white);
    glMaterialfv(GL_FRONT,  GL_SPECULAR,            white);
    glMaterialf(GL_FRONT,   GL_SHININESS,           30);

    glLightfv(GL_LIGHT0, GL_AMBIENT,  black);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glLightfv(GL_LIGHT0, GL_POSITION, direction);

    glEnable(GL_LIGHTING);                // so the renderer considers light
    glEnable(GL_LIGHT0);                  // turn LIGHT0 on
    glEnable(GL_DEPTH_TEST);              // so the renderer considers depth
    glEnable(GL_CULL_FACE);
    glDepthFunc(GL_LESS);

    window.pushGLStates();

    // --------------------------------------------
    
    srand(std::hash<int>{}(frameClock.getElapsedTime().asMicroseconds()));

    bool running = true;
    while (running) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);
            if (event.type == sf::Event::KeyPressed)
            {
                switch (event.key.code)
                {
                case sf::Keyboard::E: {
                    cursorGrabbed = !cursorGrabbed;
                    sf::Mouse::setPosition(sf::Vector2i(windowMiddle.x, windowMiddle.y), window);
                }break;
                }
            }
            else if (event.type == sf::Event::MouseMoved)
            {
                if (cursorGrabbed)
                {
                    sf::Mouse::setPosition(sf::Vector2i(windowMiddle.x, windowMiddle.y), window);
                    if (event.mouseMove.x != windowMiddle.x || event.mouseMove.y != windowMiddle.y)
                    {
                        const float sensitivity = 0.1f;

                        cameraYaw += (event.mouseMove.x - windowMiddle.x) * sensitivity;
                        cameraPitch += (event.mouseMove.y - windowMiddle.y) * -sensitivity;

                        if (cameraYaw > 180.f)
                            cameraYaw = -180.f;
                        if (cameraYaw < -180.f)
                            cameraYaw = 180.f;
                        if (cameraPitch > 89.0f)
                            cameraPitch = 89.0f;
                        if (cameraPitch < -89.0f)
                            cameraPitch = -89.0f;

                        glm::vec3 direction;
                        direction.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
                        direction.y = sin(glm::radians(cameraPitch));
                        direction.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
                        cameraFront = glm::normalize(direction);
                    }
                }
                mousePos = sf::Vector2f(event.mouseMove.x - float(windowMiddle.x), event.mouseMove.y - float(windowMiddle.y));
            }
            else if (event.type == sf::Event::Resized)
            {
                windowSize = window.getSize();
                windowMiddle = sf::Vector2i(windowSize.x / 2, windowSize.y / 2);
                //centreView.setSize(sf::Vector2f(windowSize.x, windowSize.y));
                //centreView.setCenter(0, 0);
                double ratio = double(windowSize.x) / double(windowSize.y);
                window.popGLStates();
                glViewport(0, 0, windowSize.x, windowSize.y);
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                glFrustum(-frustRight * ratio, frustRight * ratio, -frustUp, frustUp, nearClip, farClip);
                glMatrixMode(GL_MODELVIEW);
                window.pushGLStates();
            }
            else if (event.type == sf::Event::Closed) {
                running = false;
            }
        }
        const float cameraSpeed = 0.05f; // adjust accordingly
        float accelerate = (1.f + float(sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)));
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
        {
            cameraPos += cameraSpeed * cameraFront * accelerate;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
        {
            cameraPos -= cameraSpeed * cameraFront * accelerate;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
        {
            cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed * accelerate;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
        {
            cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed * accelerate;
        }
        
        window.clear();
        ImGui::SFML::Update(window, deltaClock.restart());

        float frameRate = 1000000.f / float(frameClock.getElapsedTime().asMicroseconds());
        ImGui::Begin("Update Rate");
        static int updatesPerFrame = 1;
        ImGui::Text(std::string("FPS:     " + std::to_string_with_precision(frameRate)).c_str());
        ImGui::Text(std::string("UPS:     " + std::to_string_with_precision(frameRate * float(updatesPerFrame))).c_str());
        ImGui::SliderInt(" :UPF", &updatesPerFrame, 1, 50);
        ImGui::SliderInt(" :THREADS", &NUM_THREADS, 1, 20);
        ImGui::Checkbox(" :Run Simulation", &runSimulation);
        ImGui::Checkbox(" :Draw Plane", &drawPlane);
        ImGui::Text(std::string("POS : " + std::formatted_double(cameraPos.x) + "x, " +std::formatted_double(cameraPos.z) + "y, " + std::formatted_double(cameraPos.z) + "z").c_str());
        ImGui::Text(std::string("MPOS: " + std::to_string_with_precision(mousePos.x) + "x, " + std::to_string_with_precision(mousePos.y) +"y").c_str());
        ImGui::End();
        frameClock.restart();

        //ImGui::ShowTestWindow();

        {
            window.popGLStates();

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glLoadMatrixf(glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp));

            // Draw a white grid "floor" for the tetrahedron to sit on.
            glColor3f(1.0, 1.0, 1.0);

            if (drawPlane)
            {
                glBegin(GL_LINES);
                const float lines = 50;
                const float gap = 0.5;
                float dist = lines * gap * 0.5;
                for (GLfloat i = -dist; i <= dist; i += gap) {
                    glVertex3f(i, 0, dist); glVertex3f(i, 0, -dist);
                    glVertex3f(dist, 0, i); glVertex3f(-dist, 0, i);
                }
                glEnd();
            }
            
            glCullFace(GL_FRONT);
            glFlush();
            window.pushGLStates();
        }
        //sf::ContextSettings windowSettings = window.getSettings();
        //std::cout << "windowSettings.DepthBits: " << windowSettings.depthBits << "\n";
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    
    return 0;
}
