procedure Test is
   generic
      Found : Boolean;
   procedure Find;

   procedure Find is
   begin
      Found := True;
   end Find;

   X : Boolean;

   procedure My_Find is new Find(Found => X);
begin
   My_Find;
end Ex1;
