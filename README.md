# RF*diffusion*
RFdifussion es un método para generar estructuras tridimensionales con o sin información condicional (como un motivo estructural por ejemplo). Puede ser utilizado para solventar algunos desafíos actuales en el diseño de proteínas como se destaca en el [paper original](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1).

# Instalación. ¿Cómo comenzar?

RF*difussion* está disponible como un [Google Colab Notebook](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/v1.1.1/rf/examples/diffusion.ipynb) para poder ser ejecutado desde ahí. Antes de utilizar RF*diffusion* es recomendable leer este README.

Si queremos ejectuar RF*difussion* en local, debemos seguir los siguientes pasos:

Clonar el repositorio original de github (para ello debemos tener git instalado):

````
git clone https://github.com/RosettaCommons/RFdiffusion.git
````

Después deberemos descargar los modelos en el directorio RFDiffusion:

````
cd RFdiffusion
mkdir models && cd models
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt

Optional:
wget http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt

# original structure prediction weights
wget http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt
````

También debemos asegurarnos de tener Anaconda o miniconda instalado y necesitamos instalar la implementación de NVIDIA de los [transformadores SE(3)](https://developer.nvidia.com/blog/accelerating-se3-transformers-training-using-an-nvidia-open-source-model-implementation/), que pueden ser instalados de la siguiente forma:

````
conda env create -f env/SE3nv.yml

conda activate SE3nv
cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../.. # change into the root directory of the repository
pip install -e . # install the rfdiffusion module from the root of the repository
````

Antes de correr RF*diffusion* nos deberemos asegurar de activar conda usando el siguiente comando:

````
conda activate SE3nv
````

# Uso

![main](https://github.com/user-attachments/assets/37c00dbe-b698-4ba8-80d9-5c13c13e355f)

El script que ejecutaremos se llama `run_inference.py` que se encuentra en la carpeta Scripts. Hay muchas formas de ejecutarlo, siendo la más interesante el uso de [Hydra confings](https://hydra.cc/docs/configure_hydra/intro/), que permiten especificar muchas opciones diferentes, con valores predeterminados extraídos de los modelos, de forma que la inferencia siempre, por defecto, coincide con el *entrenamiento*. Esto implica que los valores por defecto que se encuentran en `config/inference/base.yml` podrían no coincidir con los valores reales usados durante la inferecencia. Todo es manejado de manera automática.

## Ejecución básica para un monómero sin condiciones

Para una proteína de 150 aminoácidos, necesitamos especificar tres cosas:

* La longitud de la proteína
* El directorio donde queremos escribir los ficheros
* El número de modelos que queremos

````
./scripts/run_inference.py 'contigmap.contigs=[150-150]' inference.output_prefix=test_outputs/test inference.num_designs=10
````

Vamos a ver este código en detalle. En primer lugar, ¿qué es `contigmap.contigs`? Hydra congifs le dice al script como debe ser ejecutado. Para mantenerlo todo organizad, la configuracion tiene varias sub-configuraciones, siendo una de ellas el `contigmap`, que corresponde con todo lo relacionado con el string contig (que define la proteína que se va a construir). Todo lo que se encuentra en la configuración se puede sobrescribir manualmente en la línea de comandos. Podemos, por ejemplo, cambiar como funciona el difusor:

````
diffuser.crd_scale=0.5
````

... pero no es recomendable si no sabemos lo que estamos haciendo.

¿Qué significa `contigmap.contigs=[150-150]`? El string contig tiene que ser pasado como un único objecto en una lista, en vez de como string (por razones que tienen que ver con hydra) y el argumento debe estar entre `''`para que la líneas de comandos no intente analizar ninguno de los carácteres especiales. Este contig string permite especificar el rango de longitudes, pero en este caso solo queremos una proteína de 150 aminoácidos, por lo que especificamos [150-150]. 

`inference.num_designs` determina el número de trayectorias que van a ser generadas y las guarda en el directorio que especifiquemos.

La primera vez que ejecutemos RF*diffusion* estará bastante rato en "Calculating IGSO3", pero una vez se haya completado se mantendrá para futuros cálculos. Podemos encontrar más ejemplos para monómeros sin condiciones en `./examples/design_unconditional.sh`.

## Andamiaje de motivos estructurales

RF*diffusion* puee usarse para construir estructuras base o "andamios" de motivos estructurales, en lo cual destaca sobre métodos anteriormente utilizados como el Constrained Hallucination o RFjoint Inpainting.

![motif](https://github.com/user-attachments/assets/629e8b72-2c37-4339-bfe8-65dfc959637d)

Para el construir estos andamios de motivos proteicos, necesitamos una forma de especificar que queremos usar como andamio cierta proteína (uno o más segmentos de un `.pdb`) y como queremos que se conecten y en cuántos residuos en la nueva proteínea. Necesitamos ser capaces de simular diferentes formas de interacción entre las proteinas, pues en un principio no sabemos de manera precisa que residuos necesitamos.  Este trabajo está controlado por los contigs, controlados a su vez por la configuración del contigmap en hydra config. Brevemente:

* Cualquier cosa que tenga como prefijo una letra indicda que es un motivo, con la letra correspondiente a la letra de la cadena en el .pdb. Por ejemplo, A10-25 pertenece a los residuos ('A',10; 'A',11;...) en el .pdb utilizado como input
* Cualquier cosa que no tenga como prefijo una letra debe ser construido. Puede ser utilizado como un rango de longitudes. Estos rangos de longitudes pueden son simulados al azar en cada iteración de la inferencia
* Para especificar finales de cadena usamos `/0`

Si queremos usar como andamio los residuos 10 a 25 de la cadena A del `.pdb`, esto ser haría de la siguiente forma: `contigmap.contigs=[5-15/A10-25/30-40]`. De esta forma le estaremos pidiendo a RF*diffusion* que construya 5-15 residuos en el extremo N-terminal de los residuos A10-25, seguidos de 30-40 residuos más en el C-terminal. Si queremos asegurarnos que la longitud es siempre de 55 residuos, esto puede especificar con `contigmap.length=55-55`. Necesitamos proporcionar el path del archivo pdb: `inference.input_pdb=path/to/fil.pdb`. No importa que el archivo pdb tenga residuos que no queremos usar como andamio -el mapa define que residues del pdb serán usados como motivo estructural. En otras palabras, incluso si el pdb tiene una cadena B, y otros residuos en la cadena A, solo los residuos A10-25 serán usados en RF*diffusion*.

Un ejemplo puede encontrarse en `./examples/design_motifscaffolding.sh`.

### El modelo del centro activo presenta motivos muy pequeños

Los autores del programa se dieron cuenta que para motivos muy pequeños, RF*diffusion* tiende a no mantenerlos fijados en el output. Por tanto, para el andamiaje de sitios pequeños como los centros activos de las enzimas, se ajustó RF*diffusion*, permitiéndole mantener estos motivos más pequeños en su sitio y consiguiendo resultados más eficientes *in silico*. Si el motivo estructural del input es muy pequeño, los autores recommienda usar este modelo que puede especificarse usando: `inference.ckpt_override_path)models/ActiveSite_ckpt.pt`

### La flag `inpaint_seq`

La idea es que, al "fusionar" dos proteínas, los residuos que estában en la superficie de una proteína (y que normalmente son polares), deben ser ahora colocados en el "core" de la proteína, por lo que queremos que se conviertan en residuos hidrofóbicos. Lo que podemos hacer, en vez de mutar directamente estos residuos podemos enmascarar su identidad de secuencia y permitir que RF*diffusion* razone sobre ella. Esto requiere un modelo diferente que el modelo base de difusión, un modelo que ha sido entrenado para entender esta paradigma y que es controlado por el script sin necesidad de que nosotros hagamos nada. Para especificar que aminoácidos deben ser "escondidos" podemos utilizar lo siguiente:

````
contigmap.inpaint_seq=[A1/A30-40]
````

En este caso estamos enmascarando la identidad del residuo A1 y todos los residuos entre A30 y A40 (incluidos).

Un ejemplo en el que se ejecuta el andamiaje de motivos con la flag `contigmap.inpaint_seq`se encuentra en `./examples/design_motifscaffolding_inpaintseq.sh`.

### Sobre `diffuser.T`

RF*diffusion* fue entreando con 200 paso de tiempo discretos. Sin embargo, mejoras recientes han permitido reducir el número de pasos de tiempo (dt) que necesitamos para usar en la inferencia. En muchos casos, usar 20 pasos proporciona resultados equivalentes en calidad a utilizar 200 pasos (10x aumento de velocidad). Por defecto ahora se utilian 50 pasos. Esto es importante para entender la difusión parcial.

## Difusión parcial

Algo que podemos hacer con la difusión es añadir parcialmente ruido o eliminarlo de una estructura para conseguir cierta diversidad alrededor de un plegamiento. Para especificarlo debemos utilizar la entrada diffuser.parcial_T y configurar el paso de tiempo para "ruido".

https://github.com/RosettaCommons/RFdiffusion/blob/main/img/partial.png

A mayor ruido, mayor diversidad. Se recomienda utilizar diferentes valores de `diffuser.partial_T` para encontrar los mejores valores para nuestros problemas específicos. Cuando se hace una difusión parcial, en la cual estmamos difundiendo a partir de una estructura conocida, se crean ciertas restricciones. Podemos seguir utilzando contif, pero el string contig debe tener la misma longitud que la proteína que utilicemos como entrada. Por ejemplo, si consideramos un complejo entre un target y una molécula que se una a ella, y queremos diversificar la molécula (longitud 100, cadena A), deberemos hacer lo siguiente:

````
contigmap.contigs=[100-100/0 B1-150]0 diffuser.parcial_T = 20
````

La razón para esto es que si la proteína de entrada tiene solo 80 aminoácidos, pero especificamos una longitud deseada de 100, no se sabe desde donde difundir los 20 aminoácidos restantes, y por tanto, no entrarán en la distribución que RF*diffusion* ha aprendido para eliminar el ruido. Un ejemplo de difusión parcial se encuentra en `./examples/design_partialdifussion.sh'.

También podemos mantener fijas partes de la secuencia de la cadena que va a difundir. En el contecto por ejemplo de unión de un  péptido helicoidal a un target. Si hemos unido una secuencia peptídica helicoidal a una hélice ideal, y queremos diverisificar el complejo, permitiendo que la hélice sea predicha no como una hélice ideal:

````
contigmap.contigs[100-100/0 20-20] contigmap.provide_seq[100-119] diffuser.partial_T=10
````

En este caso, la cadea de 20 aminoácidos es el péptido helicoidal. La entrada `contigmpa.provide_seq` es de base 0 (índice) y puede utilizar para proporcionar un rango (100-119 es un rango inclusivo). Se pueden proporcionar múltiples rangos de secuencia separados por una coma, e.g `contigmap.provide_seq=[172-177, 200-205]`.

Un ejemplo de difusión parcial proporcionando una secuencia en las regiones que difunden se puede encontrar en `./examples/design_partialdifussion_withseq.sh`. El mismo ejemplo especificando múltiples rangos de secuencia se puede encontrar en `./examples/design_partialdifussion_multipleseq.sh`.

## Diseño de moléculas que se unen a un target

RF*diffusion* es excelente para el diseño de moléculas que se unen a un target *de novo*, también llamados *binders*. 

![binder](https://github.com/user-attachments/assets/da4b9422-249b-49d8-a308-2519a17e1ab0)

Si el target se encuentra en la cadena B:

````
./scripts/run_inference.py 'contigmap.contigs=[B1-100/0 100-100]' inference.output_prefix=test_outputs/binder_test inference.num_designs=10
````

Esto generará moléculas de 100 residuoes que se unan a los residuos 1-100 de la molécula B. 

Probablemente esta no sea la mejor forma de generar binders. La difusión es computacionalmente costosa, necesitamos probar y hacerlo lo más rápido posible. Proporcionar el target completo, sin cortar, hará que la difusión sea lento si el target es grande (y la mayoría de targets de interés, como por ejemplo receptores de superficie celular, tienden a ser grandes). Un método probado para aumentar la velocidad del desarrollo de binders es cortar el target alrededor de la zona de interacción deseada. Pero esto genera un problema! Si cortas el target puedes exponer el centro hidrofóbico que estban enterrados antes de crotar, así que, ¿cómo podemos garantizar que el binder irá a la interfase deseada en la superficie del target, y no hacia la zona hidrofóbica que se acaba de crear?

Este problema fue solventado proporcionado al modelo lo que se conoce como "residuos calientes" o *hotspots residues*. Los modelos que hemos descargado previamente han sido entrenados con hotspots, de forma que al modelo se le indicaron los residuos de algunos target que están en contacto con sus binder (los residuos que forman parte de la interfase). El modelo aprende que debe generar una interfase que implique estos hotspots. En el momento de la inferencia, podemos proporcionar los residuos para definir una región que el binder deba contactar. Se especifican de la siguiente forma: `ppi.hotspot_res=[A30,A33,A34]`, donde A es el ID de la cadena en el pdb de entrada y el número es el índice del residuo en este mismo fichero.

Se observó que el modelo general normalmente genera binders helicoidales. Estos tienen una alta tasa de acierto computacional. Sin embargo, en muchos casos otros tipos de topología podrían ser deseados. Para ello, incluimos un modelo en fase beta que genera una mayor diversidad de topologías, pero no ha sido probado experimentalmente. Lo probaremos asumiendo riesgos:

````
inference.ckpt_overrid_path=models/Complex_beta_ckpt.pt
````

Un ejemplo de diseño de binders con RF*diffusion* se encuentra en `./examples/design_ppi.sh`

## Consideraciones prácticas en el diseño de binders

### Seleccionar el sitio de unión con el target

No cualquier sito de un targe es un buen candidato para el diseño de binders. Para que un sitio sea un candidato atractivo para la unión debe tener más de 3 residuos hidrofóbicos para que el binder pueda interaccoinar. La unión a sitios polares es todavía bastante complicada. También es complicado unir binders a sitios con glicanos cerca pues estos se ordenan tras la unión y requeriría mucha energía. Históricamente, pese a que el diseño de binders también ha intentado evitar loops desorganizados, no está claro si esto es un requerimiento pues RF*diffusion* se ha utilizado para unir peptidos sin estructura que tienen muchas carácteristicas en común con loops desorganizados.

### Truncar la proteína target

El tiempo de cálculo de RF*diffusion* escala con O(N^2) donde N es el número de residuos del sistema. Por ello, es una buena idea truncar los targets grandes para que el cálculo no sea innecesariamente largo. Todos los pasos que siguen a RF*diffusion*, como el uso de AlphaFold, están diseñados para targets truncados. Esto es un arte. Para algunos targets, como membranas extracelulares, un punto natural es donde dos dominions se unen por un linker flexible. Para otras proteínas, como las proteínas víricas, este punto es menos obvio. Generalmente queremos preservar la estructura secundaria e introducir el menor número de rupturas posible. También debemos intentar dejar ~10A de proteína target a cada lado del sitio de unión esperado con el binder. Lo más recomendado es usar PyMol para truncar la proteína.

### Elegir los hotspots

Los hotspots son una característica que se integró en el modelo para permitir el control del sitio de unión del target con el binder. Un hotspot es un residuo del target que está dentro de un radio de 10A del binder. De todos los hotspots identificados en los targets de los modelos, entre el 0-20% son actualmente proporcionados por el modelo. En el momento de la inferencia el modelo está esperando hacer más contactos de los que le especifiquemos. Normalmente recomendamos especificar 3-6 hotspots, por lo que es recomendable ejectutar algunas pruebas antes de generar miles de diseños para asegurarnos que el número de hotspotss que especifiquemos de resultados.

### Escala del binder

Para algunos targets basta con generar ~1000 backbones. Lo que se busca es conseguir sugicientes diseños que pasen el filtro pAE_interaction < 10 (descrito más adelante). Diseños que no pasan este filtro no son útiles pues no funcionarán experimentalmente. 

### Diseño de secuencia de los binders

Los binders generados por RF*diffusion* presentan una secuencia de poli-G. RF*diffusion* no genera secuencias para la región deseada, por tanto, otro método debe ser usado para asignar una secuencia a los binders. En el paper original se utiliza el protoclo ProteinMPNN-FastRelax y es el más recomendado. El código para este protocolo puede encontrarse en este repositorio de [GitHub](https://github.com/nrbennet/dl_binder_design).

### Filtrado de los binders

Una de las cosas más importantes en el desarrollo de binders es el paso de filtrarlos para evaluar si realmente van a funcionar. En el paper se filtraron los binders usando AF2 y los scripts para el protocolo se pueden encontrar [aquí](https://github.com/nrbennet/dl_binder_design). Filtrar usando pae_interaction < 10 es un buen predictor para ver si un binder funcionará experimentalmente.

## Acondicionamiento de los plegamientos

Una de las particularidades que mejor funcionan es acondicionar el diseño de binders (o la generación de modelos) a topologías específicas. Esto se consigue proporcionar una estructura secundaria (parcial) e información a un modelo que ha sido entrenado para ello. 

![fold_cond](https://github.com/user-attachments/assets/0a15af75-32ab-4581-8af0-d5c1d10584f5)

Todavía estan trabajando para generar el input durante la inferencia, pero por ahora se pueden generar inputs directamente a partir de archivos pdb. Esto permite la especificación de topologías con baja resolución (es decir, si queremos por ejemplo un barril TIM pero no nos importa los residuos que contenga). En `helper_scripts/` podemos encontrar un script llamado `make_secstruc_adj.py`que puede usarse de la siguiente manera:

````
./make_secstruc_adj.py --input_pdb ./2KL8.pdb --out_dir /my/dir/for/adj_secstruct
````

````
./make_secstruc_adj.py --pdb_dir ./pdbs/ --out_dir /my/dir/for/adj_secstruct
````

Esto procesará un único pdb o una carpeta de pdbs, y generará una estructura secundaria y un fichero pytorch que podrá ser introducido en el modelo. Por ahora, también sería interesante generar estos archivos para el target (aunque no es necesario) y proporcionarlos al modelo. Se pueden usar en la inferencia de la siguiente forma:

````
./scripts/run_inference.py inference.output_prefix=./scaffold_conditioned_test/test scaffoldguided.scaffoldguided=True scaffoldguided.target_pdb=False scaffoldguided.scaffold_dir=./examples/ppi_scaffolds_subset
````

## Genera oligómeros simétricos

RF*diffusion* también sirve para el diseño de oligómeros simétricos. Esto se consigue al simetrizar el ruido que simulamos a t = T y simetrizar el input en cada paso de tiempo. Por ahora se puede hacer simetría:

* Cíclica
* Dihédrica
* Tetrahédrica

![olig2](https://github.com/user-attachments/assets/b5627015-b63d-4a9a-a675-3ee3b21b55ed)

Un ejemplo:

````
./scripts/run_inference.py --config-name symmetry  inference.symmetry=tetrahedral 'contigmap.contigs=[360]' inference.output_prefix=test_sample/tetrahedral inference.num_designs=1
````

Aquí hemos de especificar un fichero de configuración diferente (`--cofig-name symmetry`). La difusión simétrica es diferente de la difusión descrita arriba. Usando este fichero la difusión se usa en modo `symmetry-mode`.

El tipo de simetría se especifica en `inference.symmetr=`. En el ejemplo se ha usado la simetría tetrahédrica, pero también podemos seleccionar cíclica o dihédrica. 

La longitud del contingmap.contigs se refiere a la longitud total del oligómero. Por lo que debe ser divisible entre *n* cadenas.

Más ejemplos del diseño de oligómeros se pueden encontrar en: `./examples/design_cyclic_oligos.sh, ./examples/design_dihedral_oligos.sh, ./examples/design_tetrahedral_oligos.sh`.

## Usando potenciales auxiliares

## Andamiaje de motivos simétricos

También podemos combinar difusión simétrica con andamaiaje de motivos para motivos simétricos. Actualmente, solo hay una forma de hacer esto: especificnado la posición del motivo en particular y los ejes de simetría. 

![sym_motif](https://github.com/user-attachments/assets/52e2fea3-e919-4ca8-91f2-47c5e74f746f)

Se requiere que el usuario tenga una versión simetrizada del motivo en el pdb de entrada para este proceso. Hay dos razones principales para ello. Primero, el modelo está entrenado al centrar cualquier motivo en el origen y por tanto, el código centra los motivos automáticamente. Por tanto, si el motivo no está simetrizado, el centrado resultará en una unidad asimétrica que tiene el origen y los ejes de simetría mal colocados. Segundo, el código de difuisión usa un set canónico de ejs de simetría (matrices de rotación) para propagar la unidad asimétrica del motivo. Para prevenir ejecutar trayectorias de difusión que propaguen el motivo de formas indeseadas, se requiere que el usuario simetrice el input usando los ejes canónicos.

Hay un script de ejemplo en `examples/design_nickel.sh`para hacer andamiaje de los dominios de unión a Ni. Esto combina muchos conceptos discutidos previamente, incluyendo la generación de oligómeros simétricos, el andamiaje de motivos y el uso de potenciales auxiliares. 

## Notas sobre el peso de los modelos

No hay un modelo para gobernarlos a todos. Es decir, si queremos correr un acondicionamiento de estructura secundaria, esto requiere un modelo diferente. Todo es controlado de manera automática - el input es analiado y se trabaja desde el punto de control más apropiado. Por ello la configuración es tan importante. El punto de control exacto utilizado en la inferencia contiene todos los parámetros con los que se ha entrenado el modelo, por lo que podemos ajustar el archivo de configuración con esos valores. Si queremos especificar un punto de control diferente (si por ejemplo, entrenamos un nuevo modelo) tenemos que estar seguros de que es compatible con lo que queremos hacer, si no fallará.

## Entendiendo los ficheros de salida

Hay varios ficheros de salida:

1. El fichero .pdb. Este fichero es la predicción final del modelo. Cualquier residuo diseñado es generado como una glicina (pues solo se genera el backbone), y no hay cadenas laterales en el output. Esto es porque no hay pérdidas asociadas a las predicciones, por lo que no pueden ser totalmente consideradas como fiables.

2. El fichero .trb. Este fichero contiene metadata útil asociado con la ejecución, incluyendo el contig utilizado así como las configuración utilizada por RF*diffusion*. También encontramos otros objetos interesantes en este fichero:
   * detalles sobre el mapeo (es decir como los residuos en el input concuerdan con los residuos en el output)

3. Ficheros de trayectoria. Por defecto, los ficheros se geenran en la carpeta `/traj/`. Estos ficheros pueden ser abiertos en pymol, como pdbs con varios estados. Están ordeandos al revés, por lo que el primer pdb es la última (t=1) predicción hecha en la inferencia. Incluyen predicciones px0 (lo que el modelo predice en cada paso de tiempo) y trayectorias Xt-1 (lo que entra en el modelo a cada paso de tiempo). 

## Docker 

En `docker/Dockerfile`se encuentra un Dockerfile para ejecutar RF*diffusion* en HPC u otros sistemas. Para contstruir y ejecutar el contenedor debemos seguir los siguientes pasos: 

1. Clonar el respositorio con `git clone https://github.com/RosettaCommons/RFdiffusion.git` y después hacer `cd RFdiffusion`
2. Verificar que el daemon del Docker está corriendo en nuestro sistema con `docker info`.
3. Crear la imagen del contenedor en el sistema con `docker build -f docker/Dockerfile -t rfdiffusion .`
4. Crear carpateas en el sistema con `mkdir $HOME/inputs $HOME/outputs $HOME/models`
5. Descargas los modelos con `bash scripts/download_models.sh $HOME/models`
6. Descargar el fichero de prueba con `wget -P $HOME/inputs https://files.rcsb.org/view/5TPN.pdb`
7. Correr el contenedor con:
````
docker run -it --rm --gpus all \
  -v $HOME/models:$HOME/models \
  -v $HOME/inputs:$HOME/inputs \
  -v $HOME/outputs:$HOME/outputs \
  rfdiffusion \
  inference.output_prefix=$HOME/outputs/motifscaffolding \
  inference.model_directory_path=$HOME/models \
  inference.input_pdb=$HOME/inputs/5TPN.pdb \
  inference.num_designs=3 \
  'contigmap.contigs=[10-40/A163-181/10-40]'
````

Esto inicialiará el contenedor `rfdiffusion`, monta los modelos, inputs y outputs, pasa las GPUs disponibles y llama a `run_inference.py` con los parámetros especificados.
