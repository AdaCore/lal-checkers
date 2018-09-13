procedure Test is
   generic
      Found : Boolean;
   procedure Find;

   Counter : Integer := 0;

   procedure Find is
   begin
      Found := True;
      Counter := Counter + 1;
   end Find;

   X : Boolean;

   procedure My_Find is new Find(Found => X);
begin
   My_Find;
end Ex1;
