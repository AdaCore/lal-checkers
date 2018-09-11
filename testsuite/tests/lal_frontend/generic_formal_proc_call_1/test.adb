procedure Main is
   generic
      with procedure F(X : out Integer) is abstract;
   procedure Test;

   procedure Test is
      X : Integer;
   begin
      F(X);
   end;
begin
   My_Find;
end Ex1;
