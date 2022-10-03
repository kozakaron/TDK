# Buborékdinamika a hidrogéngyártásban: energetikai hatékonyság numerikus optimalizációja
## Bubble dynamics in hydrogen production: numerical optimization of energy efficiency

![plot](https://user-images.githubusercontent.com/42745647/193694372-f83b9b43-57f9-4816-a975-a2fd9fcd5ec1.png)


### Absztrakt
A szonokémia azzal foglalkozik, hogy kémiai reakciókat néhány mikron átmérőjű akusztikusan gerjesztett buborékok belsejében hozzon létre. A buborékok periodikusan tágulnak, majd hirtelen összeroppannak, miközben a nyomás több száz bart, a hőmérséklet több ezer Kelvint is elérhet, ideális környezetet biztosítva bizonyos reakcióknak. Alternatív megközelítésként az egyensúlyi buboréksugárról indított periodikus ultrahangos gerjesztés helyett használható egy jelentősen kitágított buborék szabadlengése, mely időben lecsengő rezgést eredményez, és csak egy vagy néhány jelentős összeroppanást hoz létre.
Jelen dolgozat célja a modell energiára történő optimalizálásának bemutatása, egy egyszerű példán, a hidrogén gyártásán keresztül. A kezdeti buborék valamilyen nemesgázból és vízgőzből áll, a hidrogén a víz disszociációja során jön létre. Az optimalizálás során nemcsak az egyensúlyi buborékméretet és a tágítás mértékét hangolom, hanem az irodalomban gyakran konstansnak tekintett paramétereket is állítok, úgy mint a környezeti nyomás és hőmérséklet, illetve vizsgálom a használt nemesgáz típusát, és a felületi feszültség szurfaktánssal történő módosítását is. A sokdimenziós paramétertér bejárásához kisebb felbontás esetén is több milliószor le kell futtatni a szimulációt, ez pedig rendkívül időigényes. A programokat Pythonban készítettem, a gyorsításhoz Just-In-Time (JIT) fordítót, és párhuzamosítást alkalmaztam. A dolgozatban a paramétertér teljes bejárásán kívül bemutatok más, kevésbé számításigényes globális optimalizációs stratégiákat is.
Az itt bemutatott modell egyetlen, gömb alakúnak feltételezett buborékot tartalmaz, aminek belseje homogén, így egy közönséges differenciálegyenlet rendszerrel modellezhető. A buborék radiális dinamikáját a Keller-Miksis egyenlet írja le, ami mereven viselkedik, nehezítve a numerikus megoldást. A reakciómechanizmus 11 különböző anyagot, és 29 reakciót tartalmaz, az egyenletekben szereplő közel 400 együttható az elérhető legfrissebb forrásokra támaszkodik.

### Abstract
Sonochemistry deals with creating chemical reactions inside acoustically excited bubbles a few microns in diameter. The bubbles periodically expand followed by a rapid collapse, while pressures can reach hundreds of bars and temperatures thousands of Kelvins, providing an ideal environment for certain reactions. Alternatively, instead of periodic ultrasonic excitation of the bubble starting from the equilibrium radius, a free vibration of a significantly expanded bubble can be used, which produces a decaying oscillation and only one or a few significant collapses.
The purpose of this paper is to present the optimization of the model for energy, through the example of hydrogen production. The initial bubble is composed of some noble gas and water vapour, while the hydrogen is produced by the dissociation of water. In the optimisation, I not only tune the equilibrium bubble size and the expansion rate, but also adjust parameters often considered constant in the literature, such as ambient pressure and temperature, and also examine the type of noble gas used and the modification of the surface tension by using surfactants. In order to sweep this multidimensional parameter space, the simulation has to be run millions of times, even at lower resolutions, which is extremely time-consuming. I wrote the programs in Python, using a Just-In-Time (JIT) compiler and parallelization for acceleration. Besides the bruteforce parameter sweep, I also present other, less computationally expensive global optimization strategies.
The model presented here contains a single bubble, assumed to be spherical, with a homogeneous interior, and can thus be modeled by a system of ordinary differential equations. The radial dynamics of the bubble is described by the Keller-Miksis equation, which is stiff, thus making the numerical solution difficult. The state-of-the-art reaction mechanism contains 11 different compounds with 29 reactions, and nearly 400 coefficients in the equations.

### TTartalom
* **cikk**
* **INP data extractor.ipynb**: opensmokeból táblázatok
* **full model.ipynb**: teljes mechanizmus, egyszeri plotolás
* **diffeq.py**: teljes mechanizmus, importálható
* ...


